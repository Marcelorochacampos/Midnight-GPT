//
// Created by Marcelo Campos on 14/05/25.
//

#include "request.h"

#include <boost/beast/http/string_body.hpp>
#include <string>
#include <map>
#include <sstream>

// Utility to parse query string into a map
std::map<std::string, std::string> parse_query_params(const std::string& query) {
    std::map<std::string, std::string> params;
    std::istringstream ss(query);
    std::string pair;

    while (std::getline(ss, pair, '&')) {
        auto pos = pair.find('=');
        if (pos != std::string::npos) {
            auto key = pair.substr(0, pos);
            auto val = pair.substr(pos + 1);
            params[key] = val;
        }
    }
    return params;
}