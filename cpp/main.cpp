#include "slm_engine.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " bench -m <model_path> [-n <num_tokens>]\n"
              << "Options:\n"
              << "  bench               Run benchmark mode\n"
              << "  -m <model_path>     Path to .slm model binary file (required)\n"
              << "  -n <num_tokens>     Number of tokens to generate/benchmark (default: 128)\n"
              << "  -h, --help          Show this help message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-h" || mode == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    std::string model_path = "";
    int num_tokens = 128;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-n" && i + 1 < argc) {
            num_tokens = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "[ERROR] Missing required option: -m <model_path>\n";
        print_usage(argv[0]);
        return 1;
    }

    if (num_tokens <= 0) {
        std::cerr << "[ERROR] Token count (-n) must be a positive integer.\n";
        return 1;
    }

    slm::Model model;
    if (!model.load_from_file(model_path)) {
        std::cerr << "[ERROR] Failed to load model from file: " << model_path << "\n";
        return 1;
    }

    if (mode == "bench" || mode == "-m") {
        slm::InferenceEngine engine(model);
        engine.benchmark(num_tokens);
    } else {
        std::cerr << "[ERROR] Unknown mode or argument: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
