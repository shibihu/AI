// ============================================================================
// main.cpp — CLI front-end for the SLM inference engine
//
// Usage:
//   ./slm bench  -m model.slm -n 128        Benchmark N-token generation
//   ./slm run    -m model.slm -p "Hello"     Interactive generation
//   ./slm chat   -m model.slm                Interactive REPL chat mode
//   ./slm info   -m model.slm                Print model metadata
// ============================================================================

#include "slm_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

struct Args {
    std::string mode    = "bench";
    std::string model   = "model.slm";
    std::string prompt  = "";
    int         n       = 128;
    float       temp    = 0.7f;
    float       top_p   = 0.9f;
    bool        valid   = true;
};

static void usage() {
    fprintf(stderr,
        "Usage: slm <mode> [options]\n"
        "\n"
        "Modes:\n"
        "  bench   Benchmark autoregressive generation\n"
        "  run     Generate text from a prompt\n"
        "  chat    Interactive REPL chat mode\n"
        "  info    Print model metadata\n"
        "\n"
        "Options:\n"
        "  -m <path>       Path to .slm model file   (default: model.slm)\n"
        "  -n <int>        Number of tokens to generate (bench/run/chat, default: 128)\n"
        "  -p <text>       Prompt string              (run mode)\n"
        "  -t, --temp <f>  Sampling temperature        (default: 0.7)\n"
        "  --top-p <f>     Top-p nucleus sampling      (default: 0.9)\n"
        "  -h              Show this help\n"
    );
}

static Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) { usage(); exit(1); }

    args.mode = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc)  { args.model  = argv[++i]; }
        else if (a == "-n" && i + 1 < argc) { args.n     = std::atoi(argv[++i]); }
        else if (a == "-p" && i + 1 < argc) { args.prompt = argv[++i]; }
        else if ((a == "-t" || a == "--temp") && i + 1 < argc) { args.temp   = std::atof(argv[++i]); }
        else if ((a == "--top-p" || a == "-top-p") && i + 1 < argc) { args.top_p = std::atof(argv[++i]); }
        else if (a == "-h" || a == "--help") { usage(); exit(0); }
        else {
            fprintf(stderr, "[SLM ERROR] Unknown option: %s\n", a.c_str());
            args.valid = false;
        }
    }
    return args;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static std::string sanitize_token(const std::string& token) {
    std::string decoded;
    decoded.reserve(token.size());

    for (size_t i = 0; i < token.size();) {
        // Some vocabularies store raw bytes as tokens such as <0xE2>.
        if (i + 5 < token.size() && token[i] == '<' && token[i + 1] == '0' &&
            (token[i + 2] == 'x' || token[i + 2] == 'X') &&
            token[i + 5] == '>') {
            int high = hex_value(token[i + 3]);
            int low = hex_value(token[i + 4]);
            if (high >= 0 && low >= 0) {
                decoded.push_back(static_cast<char>((high << 4) | low));
                i += 6;
                continue;
            }
        }

        decoded.push_back(token[i]);
        ++i;
    }

    // GPT-2 byte-level BPE markers: U+0120 (space) and U+010A (newline).
    std::string result;
    result.reserve(decoded.size());
    for (size_t i = 0; i < decoded.size();) {
        if (i + 1 < decoded.size() &&
            static_cast<unsigned char>(decoded[i]) == 0xC4 &&
            static_cast<unsigned char>(decoded[i + 1]) == 0xA0) {
            result.push_back(' ');
            i += 2;
        } else if (i + 1 < decoded.size() &&
                   static_cast<unsigned char>(decoded[i]) == 0xC4 &&
                   static_cast<unsigned char>(decoded[i + 1]) == 0x8A) {
            result.push_back('\n');
            i += 2;
        } else {
            result.push_back(decoded[i]);
            ++i;
        }
    }
    return result;
}

static std::vector<int> encode_text(const slm::Tokenizer& tok, int vocab_size,
                                    const std::string& text, bool include_bos = false) {
    std::vector<int> tokens;
    if (include_bos) {
        int bos_id = tok.bos();
        if (!tok.valid_id(bos_id) || bos_id >= vocab_size)
            bos_id = tok.unk();
        if (tok.valid_id(bos_id) && bos_id < vocab_size)
            tokens.push_back(bos_id);
    }

    std::string word;
    bool leading_space = false;
    bool leading_newline = false;
    auto append_word = [&]() {
        if (word.empty()) return;
        int id = tok.encode_word(word, leading_space, leading_newline);
        if (id < 0 || id >= vocab_size)
            id = tok.unk();
        if (tok.valid_id(id) && id < vocab_size) {
            tokens.push_back(id);
        }
        word.clear();
        leading_space = false;
        leading_newline = false;
    };

    for (char c : text) {
        if (c == '\n' || c == '\r') {
            append_word();
            leading_newline = true;
            leading_space = false;
        } else if (c == ' ' || c == '\t') {
            append_word();
            if (!leading_newline) leading_space = true;
        } else {
            word.push_back(c);
        }
    }
    append_word();
    return tokens;
}

// ---------------------------------------------------------------------------
// Mode: info
// ---------------------------------------------------------------------------

static void cmd_info(const Args& args) {
    slm::Model model;
    if (!model.load(args.model.c_str())) {
        fprintf(stderr, "[SLM ERROR] Failed to load model\n");
        exit(1);
    }
    const auto& c = model.config();
    fprintf(stdout,
        "Model:        %s\n"
        "Layers:       %d\n"
        "Heads:        %d  (KV heads: %d)\n"
        "d_model:      %d\n"
        "d_ff:         %d\n"
        "d_head:       %d\n"
        "Vocab size:   %d\n"
        "Max seq len:  %d\n"
        "Norm eps:     %.1e\n"
        "RoPE theta:   %.0f\n",
        args.model.c_str(),
        c.n_layers, c.n_heads, c.n_kv_heads,
        c.d_model, c.d_ff, c.d_head(),
        c.vocab_size, c.max_seq_len,
        c.norm_eps, c.rope_theta);
}

// ---------------------------------------------------------------------------
// Mode: bench
// ---------------------------------------------------------------------------

static void cmd_bench(const Args& args) {
    slm::Model model;
    if (!model.load(args.model.c_str())) {
        fprintf(stderr, "[SLM ERROR] Failed to load model\n");
        exit(1);
    }

    slm::GenConfig gc;
    gc.max_tokens = args.n;
    gc.temp       = 0.0f;   // greedy for consistent benchmarking
    gc.top_p      = 1.0f;

    slm::Generator gen(model, gc, /*seed=*/42);

    fprintf(stderr, "\n[SLM BENCH] Generating %d tokens (greedy, temp=0) ...\n", args.n);

    auto t0 = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;

    gen.generate("The", [&](int id, const std::string& piece) {
        (void)id;
        (void)piece;
        tokens_generated++;
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double tps = (elapsed > 0.0) ? tokens_generated / elapsed : 0.0;

    fprintf(stderr,
        "\n[SML BENCH] Results:\n"
        "  Tokens generated: %d\n"
        "  Wall time:        %.4f s\n"
        "  Tokens/sec:       %.2f\n"
        "  ms/token:         %.2f\n",
        tokens_generated, elapsed, tps,
        (tps > 0.0) ? 1000.0 / tps : 0.0);
}

// ---------------------------------------------------------------------------
// Mode: run
// ---------------------------------------------------------------------------

static void cmd_run(const Args& args) {
    slm::Model model;
    if (!model.load(args.model.c_str())) {
        fprintf(stderr, "[SLM ERROR] Failed to load model\n");
        exit(1);
    }

    slm::GenConfig gc;
    gc.max_tokens = args.n;
    gc.temp       = args.temp;
    gc.top_p      = args.top_p;

    slm::Generator gen(model, gc, /*seed=*/1337);

    std::string prompt = args.prompt.empty() ? "Hello" : args.prompt;
    setvbuf(stdout, nullptr, _IONBF, 0);
    fprintf(stderr, "\n[Prompt] %s\n[Output] ", prompt.c_str());
    fflush(stderr);

    auto t0 = std::chrono::high_resolution_clock::now();
    int count = 0;

    gen.generate(prompt, [&](int id, const std::string& piece) {
        (void)id;
        std::cout << sanitize_token(piece) << std::flush;
        count++;
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n\n" << std::flush;
    fprintf(stderr, "\n[SLM] %d tokens in %.3fs (%.1f tok/s)\n",
            count, elapsed,
            (elapsed > 0.0) ? count / elapsed : 0.0);
}

// ---------------------------------------------------------------------------
// Mode: chat
// ---------------------------------------------------------------------------

static void cmd_chat(const Args& args) {
    slm::Model model;
    if (!model.load(args.model.c_str())) {
        fprintf(stderr, "[SLM ERROR] Failed to load model\n");
        exit(1);
    }

    const auto& config = model.config();
    const auto& tok = model.tokenizer();

    slm::KVCache cache;
    cache.initialize(config);

    slm::GenConfig gc;
    gc.max_tokens = args.n;
    gc.temp       = args.temp;
    gc.top_p      = args.top_p;

    std::mt19937 rng(1337);
    int pos = 0;
    bool first_turn = true;

    bool has_chatml = (tok.lookup("<|im_start|>") >= 0 && tok.lookup("<|im_end|>") >= 0);
    int im_end_id = tok.lookup("<|im_end|>");
    int eos_id = tok.eos();

    setvbuf(stdout, nullptr, _IONBF, 0);

    printf("============================================================\n");
    printf(" SLM Chat REPL — Interactive Conversation Mode\n");
    printf(" Model: %s\n", args.model.c_str());
    printf(" Type 'exit' or 'quit' to end the session.\n");
    printf("============================================================\n\n");

    std::string user_input;
    while (true) {
        printf("User: ");
        fflush(stdout);

        if (!std::getline(std::cin, user_input)) {
            printf("\nExiting chat.\n");
            break;
        }

        if (user_input == "exit" || user_input == "quit" ||
            user_input == "Exit" || user_input == "Quit") {
            printf("Exiting chat.\n");
            break;
        }

        if (user_input.empty()) {
            continue;
        }

        std::string turn_prompt;
        if (has_chatml) {
            turn_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        } else {
            if (first_turn) {
                turn_prompt = "User: " + user_input + "\nAssistant: ";
            } else {
                turn_prompt = "\nUser: " + user_input + "\nAssistant: ";
            }
        }

        std::vector<int> turn_tokens = encode_text(tok, config.vocab_size, turn_prompt, first_turn);

        // Ensure context length does not exceed max_seq_len (2048 tokens).
        if (pos + static_cast<int>(turn_tokens.size()) + gc.max_tokens >= config.max_seq_len) {
            printf("\n[SLM] Maximum context length (%d tokens) reached. Resetting conversation...\n\n",
                   config.max_seq_len);
            cache.initialize(config);
            pos = 0;
            first_turn = true;
            if (has_chatml) {
                turn_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
            } else {
                turn_prompt = "User: " + user_input + "\nAssistant: ";
            }
            turn_tokens = encode_text(tok, config.vocab_size, turn_prompt, true);
        }

        first_turn = false;

        std::vector<float> logits;
        for (int token : turn_tokens) {
            logits = model.forward_cached(token, pos++, cache);
        }

        printf("Assistant: ");
        fflush(stdout);

        int generated_count = 0;
        std::string generated_text;

        while (generated_count < gc.max_tokens && pos < config.max_seq_len) {
            int next_token = slm::sample_top_p(logits.data(), static_cast<int>(logits.size()),
                                               gc.temp, gc.top_p, rng);

            if (next_token == eos_id || next_token == 0 || (has_chatml && next_token == im_end_id)) {
                break;
            }

            std::string piece = tok.decode(next_token);
            if (piece == "<|endoftext|>" || piece == "</s>" || piece == "<eos>") {
                break;
            }

            if (!has_chatml) {
                generated_text += piece;
                if (generated_text.find("User:") != std::string::npos ||
                    generated_text.find("\nUser:") != std::string::npos) {
                    break;
                }
            }

            std::cout << sanitize_token(piece) << std::flush;
            generated_count++;

            logits = model.forward_cached(next_token, pos++, cache);
        }

        printf("\n\n");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (!args.valid) { usage(); return 1; }

    if      (args.mode == "info")  cmd_info(args);
    else if (args.mode == "bench") cmd_bench(args);
    else if (args.mode == "run")   cmd_run(args);
    else if (args.mode == "chat")  cmd_chat(args);
    else {
        fprintf(stderr, "[SLM ERROR] Unknown mode '%s'\n", args.mode.c_str());
        usage();
        return 1;
    }

    return 0;
}
