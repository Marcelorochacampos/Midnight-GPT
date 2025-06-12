//
// Created by Marcelo Campos on 14/05/25.
//

#ifndef REQUEST_H
#define REQUEST_H

#include <string>
#include <map>

std::map<std::string, std::string> parse_query_params(const std::string& query);

#endif //REQUEST_H
