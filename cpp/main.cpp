// ============================================================================
// main.cpp — CLI front-end for the SLM inference engine
//
// Usage:
//   ./slm bench  -m model.slm -n 128        Benchmark N-token generation
//   ./slm run    -m model.slm -p "Hello"     Interactive generation
//   ./slm info   -m model.slm                Print model metadata
// ============================================================================

#include "slm_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
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
    float       temp    = 0.8f;
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
        "  info    Print model metadata\n"
        "\n"
        "Options:\n"
        "  -m <path>   Path to .slm model file   (default: model.slm)\n"
        "  -n <int>    Number of tokens to generate (bench, default: 128)\n"
        "  -p <text>   Prompt string              (run mode)\n"
        "  -t <float>  Sampling temperature        (default: 0.8)\n"
        "  --top-p <f> Top-p nucleus sampling      (default: 0.9)\n"
        "  -h          Show this help\n"
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
        else if (a == "-t" && i + 1 < argc) { args.temp   = std::atof(argv[++i]); }
        else if (a == "--top-p" && i + 1 < argc) { args.top_p = std::atof(argv[++i]); }
        else if (a == "-h" || a == "--help") { usage(); exit(0); }
        else {
            fprintf(stderr, "[SLM ERROR] Unknown option: %s\n", a.c_str());
            args.valid = false;
        }
    }
    return args;
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
    fprintf(stderr, "\n[Prompt] %s\n[Output] ", prompt.c_str());
    fflush(stderr);

    auto t0 = std::chrono::high_resolution_clock::now();
    int count = 0;

    gen.generate(prompt, [&](int id, const std::string& piece) {
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
        count++;
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stdout, "\n\n");
    fprintf(stderr, "\n[SLM] %d tokens in %.3fs (%.1f tok/s)\n",
            count, elapsed,
            (elapsed > 0.0) ? count / elapsed : 0.0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (!args.valid) { usage(); return 1; }

    if      (args.mode == "info") cmd_info(args);
    else if (args.mode == "bench") cmd_bench(args);
    else if (args.mode == "run")   cmd_run(args);
    else {
        fprintf(stderr, "[SLM ERROR] Unknown mode '%s'\n", args.mode.c_str());
        usage();
        return 1;
    }

    return 0;
}
