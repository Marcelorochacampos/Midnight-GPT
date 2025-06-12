//
// Created by Marcelo Campos on 13/05/25.
//

#ifndef CHAR_TOKENIZER_H
#define CHAR_TOKENIZER_H

#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace midnightnn {

class CharTokenizer {
public:
    void load_state(std::string path);
    torch::Tensor encode(std::string input);
    std::string decode(const torch::Tensor &tensors);

private:
    std::unordered_map<char, int64_t> token_id_for_char;
    std::unordered_map<int64_t, char> char_for_token_id;
};

} // midnightnn

#endif //CHAR_TOKENIZER_H
