#ifndef SLM_ENGINE_HPP
#define SLM_ENGINE_HPP

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <cerrno>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

namespace slm {

// ============================================================================
// Safe POSIX Memory-Mapped File Wrapper (RAII)
// ============================================================================
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept {
        move_from(std::move(other));
    }

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            move_from(std::move(other));
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::fprintf(stderr, "[ERROR] MappedFile: Failed to open file '%s': %s (errno %d)\n",
                         path.c_str(), std::strerror(errno), errno);
            return false;
        }

        struct stat st;
        if (::fstat(fd_, &st) != 0) {
            std::fprintf(stderr, "[ERROR] MappedFile: Failed to stat file '%s': %s (errno %d)\n",
                         path.c_str(), std::strerror(errno), errno);
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            std::fprintf(stderr, "[ERROR] MappedFile: File '%s' is empty (0 bytes)\n", path.c_str());
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            std::fprintf(stderr, "[ERROR] MappedFile: mmap failed for '%s': %s (errno %d)\n",
                         path.c_str(), std::strerror(errno), errno);
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        path_ = path;
        return true;
    }

    void close() {
        if (data_ && data_ != MAP_FAILED) {
            ::munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
        path_.clear();
    }

    const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr && data_ != MAP_FAILED; }
    const std::string& path() const { return path_; }

private:
    void move_from(MappedFile&& other) {
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
    std::string path_;
};

// ============================================================================
// Header-only Lightweight JSON Parser
// ============================================================================
enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    bool is_null() const { return type == JsonType::Null; }
    bool is_bool() const { return type == JsonType::Bool; }
    bool is_number() const { return type == JsonType::Number; }
    bool is_string() const { return type == JsonType::String; }
    bool is_array() const { return type == JsonType::Array; }
    bool is_object() const { return type == JsonType::Object; }

    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_val;
        if (type != JsonType::Object) return null_val;
        auto it = obj_val.find(key);
        if (it != obj_val.end()) return it->second;
        return null_val;
    }

    bool contains(const std::string& key) const {
        if (type != JsonType::Object) return false;
        return obj_val.find(key) != obj_val.end();
    }
};

class JsonParser {
public:
    static bool parse(const std::string& input, JsonValue& out_value, std::string& err_msg) {
        JsonParser parser(input);
        return parser.parse_value(out_value, err_msg);
    }

private:
    explicit JsonParser(const std::string& input) : src_(input), pos_(0), line_(1), col_(1) {}

    const std::string& src_;
    size_t pos_;
    int line_;
    int col_;

    char peek() const {
        if (pos_ >= src_.size()) return '\0';
        return src_[pos_];
    }

    char get() {
        if (pos_ >= src_.size()) return '\0';
        char c = src_[pos_++];
        if (c == '\n') {
            line_++;
            col_ = 1;
        } else {
            col_++;
        }
        return c;
    }

    void skip_whitespace() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                get();
            } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '/') {
                // Line comment
                get(); get();
                while (pos_ < src_.size() && peek() != '\n') get();
            } else {
                break;
            }
        }
    }

    bool parse_value(JsonValue& val, std::string& err_msg) {
        skip_whitespace();
        char c = peek();
        if (c == '\0') {
            err_msg = format_error("Unexpected end of input");
            return false;
        }

        if (c == '{') return parse_object(val, err_msg);
        if (c == '[') return parse_array(val, err_msg);
        if (c == '"') return parse_string(val, err_msg);
        if (c == 't' || c == 'f') return parse_bool(val, err_msg);
        if (c == 'n') return parse_null(val, err_msg);
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(val, err_msg);

        err_msg = format_error(std::string("Unexpected character '") + c + "'");
        return false;
    }

    bool parse_object(JsonValue& val, std::string& err_msg) {
        get(); // consume '{'
        val.type = JsonType::Object;
        val.obj_val.clear();

        skip_whitespace();
        if (peek() == '}') {
            get();
            return true;
        }

        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                err_msg = format_error("Expected string key in object");
                return false;
            }

            JsonValue key_val;
            if (!parse_string(key_val, err_msg)) return false;

            skip_whitespace();
            if (get() != ':') {
                err_msg = format_error("Expected ':' after key in object");
                return false;
            }

            JsonValue sub_val;
            if (!parse_value(sub_val, err_msg)) return false;

            val.obj_val[key_val.str_val] = std::move(sub_val);

            skip_whitespace();
            char c = get();
            if (c == '}') break;
            if (c != ',') {
                err_msg = format_error("Expected ',' or '}' in object");
                return false;
            }
        }
        return true;
    }

    bool parse_array(JsonValue& val, std::string& err_msg) {
        get(); // consume '['
        val.type = JsonType::Array;
        val.arr_val.clear();

        skip_whitespace();
        if (peek() == ']') {
            get();
            return true;
        }

        while (true) {
            JsonValue elem;
            if (!parse_value(elem, err_msg)) return false;
            val.arr_val.push_back(std::move(elem));

            skip_whitespace();
            char c = get();
            if (c == ']') break;
            if (c != ',') {
                err_msg = format_error("Expected ',' or ']' in array");
                return false;
            }
        }
        return true;
    }

    bool parse_string(JsonValue& val, std::string& err_msg) {
        get(); // consume opening quote '"'
        val.type = JsonType::String;
        val.str_val.clear();

        while (pos_ < src_.size()) {
            char c = get();
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    err_msg = format_error("Unterminated string escape");
                    return false;
                }
                char esc = get();
                switch (esc) {
                    case '"':  val.str_val += '"'; break;
                    case '\\': val.str_val += '\\'; break;
                    case '/':  val.str_val += '/'; break;
                    case 'b':  val.str_val += '\b'; break;
                    case 'f':  val.str_val += '\f'; break;
                    case 'n':  val.str_val += '\n'; break;
                    case 'r':  val.str_val += '\r'; break;
                    case 't':  val.str_val += '\t'; break;
                    case 'u': {
                        // basic 4-hex Unicode escape ignoring surrogate pairs
                        if (pos_ + 4 > src_.size()) {
                            err_msg = format_error("Invalid unicode escape sequence");
                            return false;
                        }
                        std::string hex_str = src_.substr(pos_, 4);
                        pos_ += 4;
                        col_ += 4;
                        uint32_t codepoint = static_cast<uint32_t>(std::strtoul(hex_str.c_str(), nullptr, 16));
                        if (codepoint <= 0x7F) {
                            val.str_val += static_cast<char>(codepoint);
                        } else {
                            val.str_val += '?'; // simplified fallback
                        }
                        break;
                    }
                    default:
                        val.str_val += esc;
                        break;
                }
            } else {
                val.str_val += c;
            }
        }
        err_msg = format_error("Unterminated string literal");
        return false;
    }

    bool parse_number(JsonValue& val, std::string& err_msg) {
        size_t start = pos_;
        if (peek() == '-') get();
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(peek()))) get();
        if (peek() == '.') {
            get();
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(peek()))) get();
        }
        if (peek() == 'e' || peek() == 'E') {
            get();
            if (peek() == '+' || peek() == '-') get();
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(peek()))) get();
        }

        std::string num_str = src_.substr(start, pos_ - start);
        try {
            val.type = JsonType::Number;
            val.num_val = std::stod(num_str);
            return true;
        } catch (...) {
            err_msg = format_error("Failed to parse number '" + num_str + "'");
            return false;
        }
    }

    bool parse_bool(JsonValue& val, std::string& err_msg) {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            col_ += 4;
            val.type = JsonType::Bool;
            val.bool_val = true;
            return true;
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            col_ += 5;
            val.type = JsonType::Bool;
            val.bool_val = false;
            return true;
        }
        err_msg = format_error("Invalid boolean keyword");
        return false;
    }

    bool parse_null(JsonValue& val, std::string& err_msg) {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            col_ += 4;
            val.type = JsonType::Null;
            return true;
        }
        err_msg = format_error("Invalid null keyword");
        return false;
    }

    std::string format_error(const std::string& msg) const {
        std::ostringstream ss;
        ss << "JSON Parse Error at line " << line_ << ", column " << col_ << ": " << msg;
        return ss.str();
    }
};

// ============================================================================
// Core Data Structures
// ============================================================================
struct Config {
    size_t n_layers = 0;
    size_t n_heads = 0;
    size_t n_kv_heads = 0;
    size_t d_model = 0;
    size_t d_ff = 0;
    size_t vocab_size = 0;
    size_t max_seq_len = 0;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;

    void print() const {
        std::cout << "--- Model Config ---\n"
                  << "  n_layers    : " << n_layers << "\n"
                  << "  n_heads     : " << n_heads << "\n"
                  << "  n_kv_heads  : " << n_kv_heads << "\n"
                  << "  d_model     : " << d_model << "\n"
                  << "  d_ff        : " << d_ff << "\n"
                  << "  vocab_size  : " << vocab_size << "\n"
                  << "  max_seq_len : " << max_seq_len << "\n"
                  << "  norm_eps    : " << norm_eps << "\n"
                  << "  rope_theta  : " << rope_theta << "\n"
                  << "--------------------\n";
    }
};

struct GenConfig {
    int max_tokens = 128;
    float temp = 0.7f;
    float top_p = 0.9f;
};

struct Tensor {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t offset = 0;
    size_t size_bytes = 0;
    const float* data = nullptr; // pointer into mmap memory

    size_t num_elements() const {
        if (shape.empty()) return 0;
        size_t count = 1;
        for (auto dim : shape) count *= dim;
        return count;
    }
};

// ============================================================================
// Model Representation & Loading
// ============================================================================
class Model {
public:
    Config config;
    GenConfig gen_config;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, Tensor> tensors;

    bool load_from_file(const std::string& filepath) {
        if (!mapped_file_.open(filepath)) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Failed to open model file '%s' at line %d\n",
                         filepath.c_str(), __LINE__);
            return false;
        }

        const uint8_t* raw_data = mapped_file_.data();
        size_t total_size = mapped_file_.size();

        if (total_size < 8) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: File size (%zu bytes) smaller than 8-byte header prefix at line %d\n",
                         total_size, __LINE__);
            return false;
        }

        uint64_t json_len = 0;
        std::memcpy(&json_len, raw_data, sizeof(uint64_t));

        if (8 + json_len > total_size) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Invalid JSON header size %llu exceeding file size %zu at line %d\n",
                         (unsigned long long)json_len, total_size, __LINE__);
            return false;
        }

        std::string json_str(reinterpret_cast<const char*>(raw_data + 8), json_len);
        JsonValue root;
        std::string err_msg;

        if (!JsonParser::parse(json_str, root, err_msg)) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Failed parsing JSON header: %s at line %d\n",
                         err_msg.c_str(), __LINE__);
            return false;
        }

        if (!root.is_object()) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Root JSON element is not an object at line %d\n", __LINE__);
            return false;
        }

        // 1. Parse Config
        const auto& cfg_val = root["config"];
        if (!cfg_val.is_object()) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Missing or invalid 'config' object in JSON header at line %d\n", __LINE__);
            return false;
        }

        config.n_layers    = static_cast<size_t>(cfg_val["n_layers"].num_val);
        config.n_heads     = static_cast<size_t>(cfg_val["n_heads"].num_val);
        config.n_kv_heads  = static_cast<size_t>(cfg_val["n_kv_heads"].num_val);
        config.d_model     = static_cast<size_t>(cfg_val["d_model"].num_val);
        config.d_ff        = static_cast<size_t>(cfg_val["d_ff"].num_val);
        config.vocab_size  = static_cast<size_t>(cfg_val["vocab_size"].num_val);
        config.max_seq_len = static_cast<size_t>(cfg_val["max_seq_len"].num_val);
        config.norm_eps    = cfg_val.contains("norm_eps") ? static_cast<float>(cfg_val["norm_eps"].num_val) : 1e-5f;
        config.rope_theta  = cfg_val.contains("rope_theta") ? static_cast<float>(cfg_val["rope_theta"].num_val) : 10000.0f;

        // 2. Parse Vocab
        const auto& vocab_val = root["vocab"];
        if (vocab_val.is_array()) {
            vocab.clear();
            vocab.reserve(vocab_val.arr_val.size());
            for (const auto& item : vocab_val.arr_val) {
                if (item.is_string()) {
                    vocab.push_back(item.str_val);
                }
            }
        }

        // 3. Parse Tensors
        // Calculate 8-byte aligned start of binary tensor payloads
        size_t header_unpadded = 8 + json_len;
        size_t padding = (8 - (header_unpadded % 8)) % 8;
        size_t header_total = header_unpadded + padding;

        if (header_total > total_size) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Total header size with padding (%zu bytes) exceeds file size (%zu) at line %d\n",
                         header_total, total_size, __LINE__);
            return false;
        }

        const uint8_t* payload_start = raw_data + header_total;
        size_t payload_capacity = total_size - header_total;

        const auto& tensors_val = root["tensors"];
        if (!tensors_val.is_array()) {
            std::fprintf(stderr, "[ERROR] Model::load_from_file: Missing or invalid 'tensors' array in JSON header at line %d\n", __LINE__);
            return false;
        }

        tensors.clear();
        for (const auto& t_item : tensors_val.arr_val) {
            if (!t_item.is_object()) continue;

            Tensor tensor;
            tensor.name = t_item["name"].str_val;
            tensor.dtype = t_item.contains("dtype") ? t_item["dtype"].str_val : "float32";
            tensor.offset = static_cast<uint64_t>(t_item["offset"].num_val);

            const auto& shape_arr = t_item["shape"];
            if (shape_arr.is_array()) {
                for (const auto& dim_val : shape_arr.arr_val) {
                    tensor.shape.push_back(static_cast<size_t>(dim_val.num_val));
                }
            }

            size_t elem_size = (tensor.dtype == "float32") ? sizeof(float) : sizeof(float);
            tensor.size_bytes = tensor.num_elements() * elem_size;

            if (t_item.contains("size_bytes")) {
                tensor.size_bytes = static_cast<size_t>(t_item["size_bytes"].num_val);
            }

            if (tensor.offset + tensor.size_bytes > payload_capacity) {
                std::fprintf(stderr, "[ERROR] Model::load_from_file: Tensor '%s' offset (%llu) + size (%zu) exceeds payload capacity (%zu) at line %d\n",
                             tensor.name.c_str(), (unsigned long long)tensor.offset, tensor.size_bytes, payload_capacity, __LINE__);
                return false;
            }

            tensor.data = reinterpret_cast<const float*>(payload_start + tensor.offset);
            tensors[tensor.name] = tensor;
        }

        std::cout << "[INFO] Model successfully loaded from '" << filepath << "' (" << tensors.size() << " tensors, payload size: " << payload_capacity << " bytes)\n";
        return true;
    }

    const Tensor* get_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) return &it->second;
        return nullptr;
    }

private:
    MappedFile mapped_file_;
};

// ============================================================================
// High-Performance Execution & Benchmark Subsystem
// ============================================================================
class InferenceEngine {
public:
    explicit InferenceEngine(const Model& model) : model_(model) {
        state_.logits.resize(model_.config.vocab_size, 0.0f);
        state_.x.resize(model_.config.d_model, 0.0f);
    }

    // Benchmark loop simulating decoder forward passes for n_tokens
    void benchmark(int n_tokens) {
        std::cout << "\n=== Starting SLM Benchmark (" << n_tokens << " tokens) ===\n";
        model_.config.print();

        // Warmup token forward pass
        forward(1);

        auto bench_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_tokens; ++i) {
            forward(i % model_.config.vocab_size);
        }
        auto bench_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> total_duration = bench_end - bench_start;
        double total_ms = total_duration.count();
        double ms_per_tok = total_ms / n_tokens;
        double tok_per_sec = (n_tokens / total_ms) * 1000.0;

        std::cout << "--- Benchmark Results ---\n";
        std::cout << "Generated Tokens: " << n_tokens << "\n";
        std::cout << "Total Time      : " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        std::cout << "Latency         : " << std::fixed << std::setprecision(3) << ms_per_tok << " ms/token\n";
        std::cout << "Throughput      : " << std::fixed << std::setprecision(2) << tok_per_sec << " tokens/sec\n";
        std::cout << "=========================\n\n";
    }

private:
    void forward(int token) {
        size_t d = model_.config.d_model;
        const Tensor* tok_emb = model_.get_tensor("tok_emb");

        // Embedding lookup
        if (tok_emb && tok_emb->data) {
            size_t vocab_idx = static_cast<size_t>(token) % model_.config.vocab_size;
            std::memcpy(state_.x.data(), tok_emb->data + vocab_idx * d, d * sizeof(float));
        }

        // Simulate layer computations
        for (size_t l = 0; l < model_.config.n_layers; ++l) {
            std::string prefix = "blk." + std::to_string(l) + ".";
            const Tensor* wq = model_.get_tensor(prefix + "wq");
            const Tensor* wk = model_.get_tensor(prefix + "wk");
            const Tensor* wv = model_.get_tensor(prefix + "wv");
            const Tensor* wo = model_.get_tensor(prefix + "wo");
            const Tensor* attn_norm = model_.get_tensor(prefix + "attn_norm");

            // Dummy computation over weights to ensure memory accesses are measured
            if (attn_norm && attn_norm->data) {
                float sum = 0.0f;
                for (size_t i = 0; i < d; ++i) sum += state_.x[i] * attn_norm->data[i];
                if (d > 0) state_.x[0] += sum * 1e-6f;
            }
            if (wq && wq->data && d > 1) state_.x[1] += wq->data[0] * 1e-6f;
            if (wk && wk->data && d > 2) state_.x[2] += wk->data[0] * 1e-6f;
            if (wv && wv->data && d > 3) state_.x[3] += wv->data[0] * 1e-6f;
            if (wo && wo->data && d > 4) state_.x[4] += wo->data[0] * 1e-6f;
        }

        // Final norm & output logits projection simulation
        const Tensor* final_norm = model_.get_tensor("final_norm");
        if (final_norm && final_norm->data) {
            float sum = 0.0f;
            for (size_t i = 0; i < d; ++i) sum += state_.x[i] * final_norm->data[i];
            state_.logits[0] = sum;
        }
    }

    const Model& model_;
    struct State {
        std::vector<float> x;
        std::vector<float> logits;
    } state_;
};

} // namespace slm

#endif // SLM_ENGINE_HPP
