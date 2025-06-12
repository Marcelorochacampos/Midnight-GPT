//
// Created by Marcelo Campos on 13/05/25.
//

#include "char_tokenizer.h"

#include <string>
#include <vector>
#include <fstream>

#include "nlohmann/json.hpp"

namespace midnightnn {
    void CharTokenizer::load_state(std::string path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer JSON file: " + path);
        }
        nlohmann::json state;
        file >> state;

        for ( auto& [key, value] : state["token_id_for_char"].items()) {
            this->token_id_for_char[key[0]] = value;
        }

        for ( auto& [key, value] : state["char_for_token_id"].items()) {
            int64_t token_id = std::stoll(key);
            std::string val = value.get<std::string>();
            if (!val.empty()) {
                this->char_for_token_id[token_id] = val[0];
            }
        }
    }

    torch::Tensor CharTokenizer::encode(std::string input) {
        std::vector<int64_t> output;
        for (char c : input) {
            auto it = this->token_id_for_char.find(c);
            if (it != this->token_id_for_char.end()) {
                output.push_back(it->second);
            }
        }
        return torch::tensor(output, torch::TensorOptions().dtype(torch::kInt64));
    }

    std::string CharTokenizer::decode(const torch::Tensor &tensors) {
        std::string decoded;

        auto cpu_tensors = tensors.to(torch::kCPU).to(torch::kInt64);

        auto accessor = cpu_tensors.accessor<int64_t, 1>();

        for (int64_t i = 0; i < cpu_tensors.size(0); ++i) {
            int64_t token_id = accessor[i];
            auto it = this->char_for_token_id.find(static_cast<char>(token_id));
            if (it != this->char_for_token_id.end()) {
                decoded += it->second;
            }
        }
        return decoded;
    }

} // midnightnn