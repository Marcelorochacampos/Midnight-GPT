//
// Created by Marcelo Campos on 11/05/25.
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>

#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include "nn/tokenizer/char_tokenizer.h"

#include "helper/request.h"

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = net::ip::tcp;

int main() {
    std::cout << "Bootstrapping server." << std::endl;
    net::io_context ioc;
    tcp::acceptor acceptor{ioc, {tcp::v4(), 5555}};
    std::cout << "SSE server running at http://localhost:5555/sse\n";

    midnightnn::CharTokenizer tokenizer;
    tokenizer.load_state("../src/nn/tokenizer/tokenizer_state.json");
    std::cout << "Loaded tokenizer from ../src/nn/tokenizer/tokenizer_state.json" << std::endl;
    torch::jit::script::Module model;
    model = torch::jit::load("../src/nn/model/midnight_gpt_6000_01_cpu.pt", torch::kCPU);
    std::cout << "Loaded model from ../src/nn/model/midnight_gpt_6000_01_cpu.pt" << std::endl;

    try {
        while (true) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            beast::tcp_stream stream(std::move(socket));

            beast::flat_buffer buffer;
            http::request<http::string_body> req;
            http::read(stream, buffer, req);

            std::cout << "Target " << req.target() << std::endl;
            if (req.method() == http::verb::options) {
                http::response<http::string_body> res{http::status::no_content, req.version()};
                res.set(http::field::access_control_allow_origin, "*");
                res.set(http::field::access_control_allow_methods, "GET, POST, OPTIONS");
                res.set(http::field::access_control_allow_headers, "Content-Type");
                res.prepare_payload();
                http::write(stream, res);
                stream.socket().shutdown(tcp::socket::shutdown_send);
                continue; // or return; to skip the rest of the loop
            }

            if (req.target().find("/sse") != boost::beast::string_view::npos) {
                std::string target = std::string(req.target()); // e.g. "/events?name=alice&id=42"
                auto pos = target.find('?');
                std::string query = target.substr(pos + 1);
                auto params = parse_query_params(query);

                std::string message_id = params["id"];
                std::string prompt = params["prompt"];
                int64_t max_tokens = std::stoi(params["max_tokens"]);


                torch::Tensor tensors = tokenizer.encode(prompt);
                tensors = tensors.unsqueeze(0);

                torch::Tensor output_ids = tensors.clone();
                torch::Tensor context_window = tensors.clone();

                std::string header =
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "Connection: keep-alive\r\n"
                        "Access-Control-Allow-Origin: *\r\n\r\n";

                boost::asio::write(stream, boost::asio::buffer(header));


                for (int i = 0; i < max_tokens; ++i) {
                    if (context_window.size(1) >= 256) {
                        context_window = context_window.index({
                            torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)
                        });
                    }

                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(context_window);
                    torch::NoGradGuard no_grad;
                    at::Tensor logits = model.forward(inputs).toTensor();
                    logits = logits.select(1, logits.size(1) - 1);

                    at::Tensor probs = torch::softmax(logits, /*dim=*/-1);
                    at::Tensor next_token_id = torch::multinomial(probs, /*num_samples=*/1); // Shape: [1]

                    output_ids = torch::cat({output_ids, next_token_id}, /*dim=*/1);
                    context_window = torch::cat({context_window, next_token_id}, /*dim=*/1);

                    auto decoded = tokenizer.decode(next_token_id.flatten());

                    std::stringstream ss;

                    nlohmann::json response = nlohmann::json::object();
                    response["id"] = message_id;
                    response["message"] = decoded;

                    std::string message = "data: " + decoded + "\n\n";
                    std::cout << message << std::endl;
                    boost::system::error_code ec;
                    boost::asio::write(stream, boost::asio::buffer(message), ec);
                    if (ec) {
                        std::cerr << "SSE stream write error: " << ec.message() << std::endl;
                        break;
                    }
                }
                std::string message = "data: [DONE]\n\n";
                std::cout << message << std::endl;
                boost::system::error_code ec;
                boost::asio::write(stream, boost::asio::buffer(message), ec);
                if (ec) {
                    std::cerr << "SSE stream write error: " << ec.message() << std::endl;
                    break;
                }
            } else {
                http::response<http::string_body> res{
                    http::status::not_found, req.version()
                };
                res.set(http::field::content_type, "text/plain");
                res.body() = "404 Not Found";
                res.prepare_payload();
                http::write(stream, res);
            }

            stream.socket().shutdown(tcp::socket::shutdown_send);
        }
    } catch (std::exception &e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}
