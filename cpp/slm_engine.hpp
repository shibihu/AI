#pragma once
// ============================================================================
// slm_engine.hpp — Lightweight C++17 Inference Engine for Small Language Models
// Single-file / header-only. Zero external dependencies beyond POSIX + STL.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace slm {

// ============================================================================
// Diagnostics
// ============================================================================

[[noreturn]] inline void die(const char* file, int line, const char* msg) {
    fprintf(stderr, "[SLM ERROR] %s:%d: %s\n", file, line, msg);
    std::abort();
}
#define SLM_DIE(msg) ::slm::die(__FILE__, __LINE__, (msg))

[[noreturn]] inline void die_fmt(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[SLM ERROR] %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    std::abort();
}
#define SLM_DIE_FMT(fmt, ...) ::slm::die_fmt(__FILE__, __LINE__, (fmt), __VA_ARGS__)

inline void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[SLM INFO] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// ============================================================================
// MappedFile — Safe POSIX mmap wrapper for read-only file access
// ============================================================================

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& o) noexcept
        : data_(o.data_), size_(o.size_), fd_(o.fd_) {
        o.data_ = nullptr;
        o.size_ = 0;
        o.fd_ = -1;
    }

    bool open(const char* path) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "[SLM ERROR] Cannot open '%s': %s\n", path, strerror(errno));
            return false;
        }

        struct stat st{};
        if (fstat(fd_, &st) < 0) {
            fprintf(stderr, "[SLM ERROR] fstat('%s') failed: %s\n", path, strerror(errno));
            ::close(fd_); fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            fprintf(stderr, "[SLM ERROR] File '%s' is empty\n", path);
            ::close(fd_); fd_ = -1;
            return false;
        }

        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "[SLM ERROR] mmap failed for '%s' (%zu bytes): %s\n",
                    path, size_, strerror(errno));
            ::close(fd_); fd_ = -1;
            return false;
        }
        data_ = static_cast<const uint8_t*>(p);
        return true;
    }

    void close() {
        if (data_ && data_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
        }
        if (fd_ >= 0) ::close(fd_);
        data_ = nullptr;
        size_ = 0;
        fd_ = -1;
    }

    const uint8_t* data()                const { return data_; }
    size_t         size()                const { return size_; }
    const char*    cstr()                const { return reinterpret_cast<const char*>(data_); }

    const uint8_t* offset(size_t bytes)  const {
        if (bytes > size_) SLM_DIE("MappedFile::offset out of bounds");
        return data_ + bytes;
    }
    const float* as_float(size_t byte_off = 0) const {
        return reinterpret_cast<const float*>(offset(byte_off));
    }

private:
    const uint8_t* data_ = nullptr;
    size_t         size_ = 0;
    int            fd_   = -1;
};

// ============================================================================
// Lightweight JSON Value & Recursive-Descent Parser (self-contained)
// ============================================================================

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;

    bool        bool_val = false;
    double      num_val  = 0.0;
    std::string str_val;
    std::vector<JsonValue>                    arr_val;
    std::unordered_map<std::string, JsonValue> obj_val;

    JsonValue() = default;
    JsonValue(std::nullptr_t) : type(Null) {}
    JsonValue(bool v)         : type(Bool),   bool_val(v) {}
    JsonValue(double v)       : type(Number), num_val(v) {}
    JsonValue(int v)          : type(Number), num_val(static_cast<double>(v)) {}
    JsonValue(size_t v)       : type(Number), num_val(static_cast<double>(v)) {}
    JsonValue(const std::string& v) : type(String), str_val(v) {}
    JsonValue(const char* v)        : type(String), str_val(v) {}

    bool is_null()   const { return type == Null;   }
    bool is_bool()   const { return type == Bool;   }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array()  const { return type == Array;  }
    bool is_object() const { return type == Object; }

    bool        as_bool()   const { return bool_val; }
    double      as_number() const { return num_val;  }
    int         as_int()    const { return static_cast<int>(num_val); }
    float       as_float()  const { return static_cast<float>(num_val); }
    const std::string& as_string() const { return str_val; }
    const std::vector<JsonValue>&   as_array()  const { return arr_val; }
    const std::unordered_map<std::string, JsonValue>& as_object() const { return obj_val; }

    double number(double def = 0.0) const {
        if (is_number()) return num_val;
        if (is_string()) {
            char* end = nullptr;
            double v = std::strtod(str_val.c_str(), &end);
            if (end != str_val.c_str() && *end == '\0') return v;
        }
        return def;
    }
    int integer(int def = 0) const {
        return static_cast<int>(number(static_cast<double>(def)));
    }

    const JsonValue& operator[](const char* k) const {
        static const JsonValue kNull{nullptr};
        if (!is_object()) return kNull;
        auto it = obj_val.find(k);
        return it != obj_val.end() ? it->second : kNull;
    }
    const JsonValue& operator[](const std::string& k) const { return (*this)[k.c_str()]; }
    const JsonValue& at(size_t i) const { return arr_val.at(i); }
};

class JsonParser {
public:
    JsonValue parse(const char* s) {
        src_ = s;
        pos_ = 0;
        JsonValue v = parse_value();
        skip_ws();
        if (src_[pos_] != '\0' && src_[pos_] != '\0') {
            // allow trailing content only if it is the null terminator
        }
        return v;
    }

private:
    const char* src_ = "";
    size_t      pos_ = 0;

    char peek()  const { return src_[pos_]; }
    char advance() { return src_[pos_++]; }
    bool at_end() const { return src_[pos_] == '\0'; }

    void skip_ws() {
        while (!at_end() && (peek() == ' ' || peek() == '\t' ||
                             peek() == '\n' || peek() == '\r'))
            advance();
    }

    bool match(char c) {
        skip_ws();
        if (peek() == c) { advance(); return true; }
        return false;
    }

    void expect(char c) {
        skip_ws();
        if (peek() != c) {
            SLM_DIE_FMT("JSON: expected '%c' at pos %zu, got '%c'",
                         c, pos_, peek());
        }
        advance();
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        SLM_DIE_FMT("JSON: unexpected char '%c' at pos %zu", c, pos_);
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue obj;
        obj.type = JsonValue::Object;
        skip_ws();
        if (peek() == '}') { advance(); return obj; }
        for (;;) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            obj.obj_val[std::move(key)] = parse_value();
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect('}');
        return obj;
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue arr;
        arr.type = JsonValue::Array;
        skip_ws();
        if (peek() == ']') { advance(); return arr; }
        for (;;) {
            arr.arr_val.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect(']');
        return arr;
    }

    std::string parse_string() {
        skip_ws();
        if (peek() != '"')
            SLM_DIE_FMT("JSON: expected '\"' at pos %zu", pos_);
        advance();
        std::string r;
        for (;;) {
            if (at_end()) SLM_DIE("JSON: unterminated string");
            char c = advance();
            if (c == '"') break;
            if (c == '\\') {
                if (at_end()) SLM_DIE("JSON: unterminated escape");
                char e = advance();
                switch (e) {
                    case '"':  r += '"';  break;
                    case '\\': r += '\\'; break;
                    case '/':  r += '/';  break;
                    case 'n':  r += '\n'; break;
                    case 'r':  r += '\r'; break;
                    case 't':  r += '\t'; break;
                    case 'b':  r += '\b'; break;
                    case 'f':  r += '\f'; break;
                    case 'u': {
                        char h[5] = {};
                        for (int i = 0; i < 4 && !at_end(); i++) h[i] = advance();
                        unsigned cp = static_cast<unsigned>(strtoul(h, nullptr, 16));
                        if (cp < 0x80) {
                            r += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            r += static_cast<char>(0xC0 | (cp >> 6));
                            r += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            r += static_cast<char>(0xE0 | (cp >> 12));
                            r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            r += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: r += e; break;
                }
            } else {
                r += c;
            }
        }
        return r;
    }

    JsonValue parse_number() {
        skip_ws();
        size_t start = pos_;
        if (peek() == '-') advance();
        while (!at_end() && peek() >= '0' && peek() <= '9') advance();
        if (!at_end() && peek() == '.') {
            advance();
            while (!at_end() && peek() >= '0' && peek() <= '9') advance();
        }
        if (!at_end() && (peek() == 'e' || peek() == 'E')) {
            advance();
            if (!at_end() && (peek() == '+' || peek() == '-')) advance();
            while (!at_end() && peek() >= '0' && peek() <= '9') advance();
        }
        std::string ns(src_ + start, pos_ - start);
        char* end = nullptr;
        double v = std::strtod(ns.c_str(), &end);
        return JsonValue(v);
    }

    JsonValue parse_bool() {
        if (std::strncmp(src_ + pos_, "true", 4) == 0) {
            pos_ += 4; return JsonValue(true);
        }
        if (std::strncmp(src_ + pos_, "false", 5) == 0) {
            pos_ += 5; return JsonValue(false);
        }
        SLM_DIE("JSON: invalid boolean literal");
    }

    JsonValue parse_null() {
        if (std::strncmp(src_ + pos_, "null", 4) == 0) {
            pos_ += 4; return JsonValue(nullptr);
        }
        SLM_DIE("JSON: invalid null literal");
    }
};

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    int   n_layers    = 0;
    int   n_heads     = 0;
    int   n_kv_heads  = 0;
    int   d_model     = 0;
    int   d_ff        = 0;
    int   vocab_size  = 0;
    int   max_seq_len = 0;
    float norm_eps    = 1e-5f;
    float rope_theta  = 10000.0f;

    int d_head() const { return n_heads > 0 ? d_model / n_heads : 0; }

    void validate() const {
        if (n_layers <= 0)    SLM_DIE("config: n_layers must be > 0");
        if (n_heads <= 0)     SLM_DIE("config: n_heads must be > 0");
        if (n_kv_heads <= 0)  SLM_DIE("config: n_kv_heads must be > 0");
        if (d_model <= 0)     SLM_DIE("config: d_model must be > 0");
        if (d_ff <= 0)        SLM_DIE("config: d_ff must be > 0");
        if (vocab_size <= 0)  SLM_DIE("config: vocab_size must be > 0");
        if (max_seq_len <= 0) SLM_DIE("config: max_seq_len must be > 0");
        if (n_heads % n_kv_heads != 0)
            SLM_DIE("config: n_heads must be divisible by n_kv_heads");
    }
};

struct GenConfig {
    int   max_tokens = 128;
    float temp       = 0.8f;
    float top_p      = 0.9f;
};

// ============================================================================
// Tensor Metadata
// ============================================================================

struct TensorInfo {
    std::string name;
    std::string dtype;    // "f32"
    std::vector<int> shape;
    size_t offset = 0;    // byte offset from tensor-data section start

    size_t num_elements() const {
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    size_t byte_size() const {
        size_t esz = 4; // f32
        if (dtype == "f16") esz = 2;
        else if (dtype == "f32") esz = 4;
        else if (dtype == "f64") esz = 8;
        return num_elements() * esz;
    }
};

// ============================================================================
// Tokenizer — simple wordpiece-like vocab from JSON
// ============================================================================

class Tokenizer {
public:
    void load(const JsonValue& vocab_obj) {
        tok2id_.clear();
        id2tok_.clear();
        if (!vocab_obj.is_object()) {
            SLM_DIE("Tokenizer: vocab must be a JSON object {token: id}");
        }
        for (auto& [k, v] : vocab_obj.as_object()) {
            int id = v.integer(-1);
            if (id < 0) SLM_DIE_FMT("Tokenizer: bad id for token '%s'", k.c_str());
            tok2id_[k] = id;
            if (static_cast<int>(id2tok_.size()) <= id)
                id2tok_.resize(static_cast<size_t>(id) + 1);
            id2tok_[static_cast<size_t>(id)] = k;
        }
        bos_id_ = lookup("<bos>");
        eos_id_ = lookup("<eos>");
    }

    int lookup(const std::string& tok) const {
        auto it = tok2id_.find(tok);
        return it != tok2id_.end() ? it->second : -1;
    }

    int encode(const std::string& tok) const {
        int id = lookup(tok);
        if (id < 0) SLM_DIE_FMT("Tokenizer: unknown token '%s'", tok.c_str());
        return id;
    }

    std::string decode(int id) const {
        if (id < 0 || id >= static_cast<int>(id2tok_.size())) return "<unk>";
        return id2tok_[static_cast<size_t>(id)];
    }

    int bos() const { return bos_id_; }
    int eos() const { return eos_id_; }

private:
    std::unordered_map<std::string, int> tok2id_;
    std::vector<std::string>             id2tok_;
    int bos_id_ = 0;
    int eos_id_ = 1;
};

// ============================================================================
// Model — loads .slm file, resolves tensor pointers
// ============================================================================

class Model {
public:
    bool load(const char* path) {
        log_info("Loading model from '%s' ...", path);

        if (!file_.open(path)) return false;

        // ---- Parse JSON header ----
        // The file layout:  [JSON header][0x00 separator][tensor data]
        // Find the null-terminated JSON region.
        size_t json_len = 0;
        while (json_len < file_.size() && file_.data()[json_len] != 0)
            ++json_len;
        if (json_len == 0 || json_len >= file_.size()) {
            SLM_DIE_FMT("Model: cannot find JSON header in '%s'", path);
        }

        JsonParser parser;
        JsonValue root = parser.parse(file_.cstr());

        // ---- Config ----
        const JsonValue& cfg = root["config"];
        if (!cfg.is_object()) SLM_DIE("Model: missing 'config' object");

        config_.n_layers    = cfg["n_layers"].integer();
        config_.n_heads     = cfg["n_heads"].integer();
        config_.n_kv_heads  = cfg["n_kv_heads"].integer();
        config_.d_model     = cfg["d_model"].integer();
        config_.d_ff        = cfg["d_ff"].integer();
        config_.vocab_size  = cfg["vocab_size"].integer();
        config_.max_seq_len = cfg["max_seq_len"].integer();
        config_.norm_eps    = cfg["norm_eps"].as_float();
        config_.rope_theta  = cfg["rope_theta"].as_float();
        config_.validate();

        log_info("Config: layers=%d heads=%d kv_heads=%d d_model=%d d_ff=%d "
                 "vocab=%d max_seq=%d eps=%.1e theta=%.0f",
                 config_.n_layers, config_.n_heads, config_.n_kv_heads,
                 config_.d_model, config_.d_ff, config_.vocab_size,
                 config_.max_seq_len, config_.norm_eps, config_.rope_theta);

        // ---- Vocabulary ----
        tokenizer_.load(root["vocab"]);

        // ---- Tensor metadata ----
        const JsonValue& tensors_json = root["tensors"];
        if (!tensors_json.is_array()) SLM_DIE("Model: missing 'tensors' array");

        tensor_data_offset_ = static_cast<size_t>(root["tensor_data_offset"].integer());
        if (tensor_data_offset_ == 0) SLM_DIE("Model: missing 'tensor_data_offset'");

        tensors_.clear();
        name2tensor_.clear();
        for (auto& tj : tensors_json.as_array()) {
            TensorInfo ti;
            ti.name   = tj["name"].as_string();
            ti.dtype  = tj["dtype"].as_string();
            const auto& shape_arr = tj["shape"].as_array();
            ti.shape.reserve(shape_arr.size());
            for (auto& s : shape_arr) ti.shape.push_back(s.integer());
            ti.offset = static_cast<size_t>(tj["offset"].integer());
            name2tensor_[ti.name] = tensors_.size();
            tensors_.push_back(std::move(ti));
        }

        // ---- Resolve weight pointers ----
        const float* base = file_.as_float(tensor_data_offset_);

        auto ptr = [&](const std::string& name) -> const float* {
            auto it = name2tensor_.find(name);
            if (it == name2tensor_.end()) {
                SLM_DIE_FMT("Model: missing tensor '%s'", name.c_str());
            }
            const TensorInfo& ti = tensors_[it->second];
            if (ti.dtype != "f32")
                SLM_DIE_FMT("Model: tensor '%s' has dtype '%s' (only f32 supported)",
                             name.c_str(), ti.dtype.c_str());
            return base + ti.offset / sizeof(float);
        };

        tok_emb_ = ptr("tok_emb");
        final_norm_ = ptr("final_norm");

        layer_attn_norm_.resize(config_.n_layers);
        layer_wq_.resize(config_.n_layers);
        layer_wk_.resize(config_.n_layers);
        layer_wv_.resize(config_.n_layers);
        layer_wo_.resize(config_.n_layers);
        layer_ffn_norm_.resize(config_.n_layers);
        layer_w1_.resize(config_.n_layers);
        layer_w2_.resize(config_.n_layers);
        layer_w3_.resize(config_.n_layers);

        for (int i = 0; i < config_.n_layers; i++) {
            std::string pfx = "layers." + std::to_string(i) + ".";
            layer_attn_norm_[i] = ptr(pfx + "attn_norm");
            layer_wq_[i]        = ptr(pfx + "wq");
            layer_wk_[i]        = ptr(pfx + "wk");
            layer_wv_[i]        = ptr(pfx + "wv");
            layer_wo_[i]        = ptr(pfx + "wo");
            layer_ffn_norm_[i]  = ptr(pfx + "ffn_norm");
            layer_w1_[i]        = ptr(pfx + "w1");
            layer_w2_[i]        = ptr(pfx + "w2");
            layer_w3_[i]        = ptr(pfx + "w3");
        }

        log_info("Model loaded successfully (%zu tensors, %zu bytes tensor data)",
                 tensors_.size(),
                 file_.size() - tensor_data_offset_);
        return true;
    }

    const ModelConfig&  config()   const { return config_;   }
    const Tokenizer&    tokenizer() const { return tokenizer_; }

    // ---- Forward pass (returns logits for the last token position) ----
    std::vector<float> forward(const std::vector<int>& tokens) const;

private:
    MappedFile  file_;
    ModelConfig config_;
    Tokenizer   tokenizer_;

    std::vector<TensorInfo>       tensors_;
    size_t                        tensor_data_offset_ = 0;
    std::unordered_map<std::string, size_t> name2tensor_;

    const float* tok_emb_    = nullptr;
    const float* final_norm_ = nullptr;

    std::vector<const float*> layer_attn_norm_;
    std::vector<const float*> layer_wq_;
    std::vector<const float*> layer_wk_;
    std::vector<const float*> layer_wv_;
    std::vector<const float*> layer_wo_;
    std::vector<const float*> layer_ffn_norm_;
    std::vector<const float*> layer_w1_;
    std::vector<const float*> layer_w2_;
    std::vector<const float*> layer_w3_;
};

// ============================================================================
// Linear Algebra Primitives
// ============================================================================

// out[d_out] = W[d_out, d_in] @ x[d_in] + b[d_out]  (row-major W, bias optional)
inline void linear(const float* x, const float* W, const float* b,
                   float* out, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        const float* row = W + static_cast<size_t>(i) * d_in;
        for (int j = 0; j < d_in; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

inline void linear(const float* x, const float* W,
                   float* out, int d_in, int d_out) {
    linear(x, W, nullptr, out, d_in, d_out);
}

// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
inline void rms_norm(const float* x, const float* w, float* out, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

// Softmax in-place
inline void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - mx);
        s += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// SwiGLU: out[i] = silu(gate[i]) * up[i]
inline void swiglu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float s = g / (1.0f + std::exp(-g));   // SiLU(x) = x * sigmoid(x)
        out[i] = s * up[i];
    }
}

// Rotate half for RoPE
// q rotates pairs, k rotates pairs.  head_stride = d_head
inline void rope_apply(float* q, float* k, int n_heads, int n_kv_heads,
                       int d_head, int pos, float theta) {
    const int half = d_head / 2;
    auto rotate = [&](float* ptr, int n_heads_this) {
        for (int h = 0; h < n_heads_this; h++) {
            float* head = ptr + h * d_head;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / std::pow(theta,
                    static_cast<float>(2 * i) / static_cast<float>(d_head));
                float t = static_cast<float>(pos) * freq;
                float c = std::cos(t);
                float s = std::sin(t);
                float a = head[i];
                float b = head[i + half];
                head[i]       = a * c - b * s;
                head[i + half] = a * s + b * c;
            }
        }
    };
    rotate(q, n_heads);
    rotate(k, n_kv_heads);
}

// ============================================================================
// Transformer Forward Pass
// ============================================================================

inline std::vector<float> Model::forward(const std::vector<int>& tokens) const {
    const int T     = static_cast<int>(tokens.size());
    const int D     = config_.d_model;
    const int H     = config_.n_heads;
    const int KH    = config_.n_kv_heads;
    const int DH    = config_.d_head();
    const int FF    = config_.d_ff;
    const int V     = config_.vocab_size;
    const float eps = config_.norm_eps;
    const float thr  = config_.rope_theta;

    // Pre-allocate work buffers
    std::vector<float> x(static_cast<size_t>(T) * D);
    std::vector<float> h(static_cast<size_t>(T) * D);
    std::vector<float> q(static_cast<size_t>(T) * H * DH);
    std::vector<float> k(static_cast<size_t>(T) * KH * DH);
    std::vector<float> v(static_cast<size_t>(T) * KH * DH);
    std::vector<float> attn_out(static_cast<size_t>(T) * D);
    std::vector<float> proj(static_cast<size_t>(T) * D);
    std::vector<float> ff_gate(static_cast<size_t>(T) * FF);
    std::vector<float> ff_up(static_cast<size_t>(T) * FF);
    std::vector<float> ff_down(static_cast<size_t>(T) * FF);

    // 1. Token embeddings:  x[t,:] = tok_emb[tokens[t]]
    for (int t = 0; t < T; t++) {
        std::memcpy(&x[static_cast<size_t>(t) * D],
                     tok_emb_ + tokens[static_cast<size_t>(t)] * D,
                     D * sizeof(float));
    }

    // 2. Transformer layers
    const int kv_rep = H / KH;   // GQA repeat factor

    for (int l = 0; l < config_.n_layers; l++) {
        // --- Attention layer norm ---
        for (int t = 0; t < T; t++) {
            rms_norm(&x[static_cast<size_t>(t) * D],
                     layer_attn_norm_[l],
                     &h[static_cast<size_t>(t) * D], D, eps);
        }

        // --- Q, K, V projections ---
        for (int t = 0; t < T; t++) {
            const float* ht = &h[static_cast<size_t>(t) * D];
            linear(ht, layer_wq_[l], &q[static_cast<size_t>(t) * H * DH], D, H * DH);
            linear(ht, layer_wk_[l], &k[static_cast<size_t>(t) * KH * DH], D, KH * DH);
            linear(ht, layer_wv_[l], &v[static_cast<size_t>(t) * KH * DH], D, KH * DH);
        }

        // --- RoPE ---
        for (int t = 0; t < T; t++) {
            rope_apply(&q[static_cast<size_t>(t) * H * DH],
                       &k[static_cast<size_t>(t) * KH * DH],
                       H, KH, DH, t, thr);
        }

        // --- Scaled dot-product attention with causal mask ---
        const float scale = 1.0f / std::sqrt(static_cast<float>(DH));
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                int kv_h = h / kv_rep;
                // Compute attention scores for this head at position t
                std::vector<float> scores(static_cast<size_t>(T));
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 > t) {
                        scores[static_cast<size_t>(t2)] = -1e9f;
                    } else {
                        float dot = 0.0f;
                        const float* qh = &q[static_cast<size_t>(t) * H * DH + h * DH];
                        const float* kh = &k[static_cast<size_t>(t2) * KH * DH + kv_h * DH];
                        for (int i = 0; i < DH; i++) dot += qh[i] * kh[i];
                        scores[static_cast<size_t>(t2)] = dot * scale;
                    }
                }
                softmax(scores.data(), T);

                // Weighted sum of values
                float* out_h = &attn_out[static_cast<size_t>(t) * D + h * DH];
                for (int i = 0; i < DH; i++) {
                    float s = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        s += scores[static_cast<size_t>(t2)] *
                             v[static_cast<size_t>(t2) * KH * DH + kv_h * DH + i];
                    }
                    out_h[i] = s;
                }
            }
        }

        // --- Output projection ---
        for (int t = 0; t < T; t++) {
            linear(&attn_out[static_cast<size_t>(t) * D],
                   layer_wo_[l],
                   &proj[static_cast<size_t>(t) * D], D, D);
        }

        // --- Residual ---
        for (size_t i = 0, n = static_cast<size_t>(T) * D; i < n; i++)
            x[i] += proj[i];

        // --- FFN layer norm ---
        for (int t = 0; t < T; t++) {
            rms_norm(&x[static_cast<size_t>(t) * D],
                     layer_ffn_norm_[l],
                     &h[static_cast<size_t>(t) * D], D, eps);
        }

        // --- FFN: gate_proj(W1) -> SiLU, up_proj(W3), SwiGLU, down_proj(W2) ---
        for (int t = 0; t < T; t++) {
            size_t bt = static_cast<size_t>(t);
            const float* ht = &h[bt * D];
            linear(ht, layer_w1_[l], &ff_gate[bt * FF], D, FF);
            linear(ht, layer_w3_[l], &ff_up[bt * FF],   D, FF);
            swiglu(&ff_gate[bt * FF], &ff_up[bt * FF], &ff_down[bt * FF], FF);
            linear(&ff_down[bt * FF], layer_w2_[l], &proj[bt * D], FF, D);
        }

        // --- Residual ---
        for (size_t i = 0, n = static_cast<size_t>(T) * D; i < n; i++)
            x[i] += proj[i];
    }

    // 3. Final RMSNorm
    std::vector<float> last_hidden(D);
    rms_norm(&x[static_cast<size_t>(T - 1) * D], final_norm_,
             last_hidden.data(), D, eps);

    // 4. Output logits via weight tying:  logits = last_hidden @ tok_emb^T
    std::vector<float> logits(V);
    for (int vi = 0; vi < V; vi++) {
        float dot = 0.0f;
        const float* row = tok_emb_ + vi * D;
        for (int j = 0; j < D; j++) dot += last_hidden[j] * row[j];
        logits[static_cast<size_t>(vi)] = dot;
    }

    return logits;
}

// ============================================================================
// Sampling
// ============================================================================

inline int argmax(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// Temperature + top-p sampling
inline int sample_top_p(const float* logits, int n, float temp, float top_p,
                        std::mt19937& rng) {
    if (temp <= 0.0001f) return argmax(logits, n);

    // Apply temperature
    std::vector<float> probs(static_cast<size_t>(n));
    float mx = *std::max_element(logits, logits + n);
    for (int i = 0; i < n; i++)
        probs[static_cast<size_t>(i)] = std::exp((logits[i] - mx) / temp);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (int i = 0; i < n; i++) probs[static_cast<size_t>(i)] /= sum;

    // Top-p filtering: sort descending, accumulate, zero out tail
    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += probs[idx[static_cast<size_t>(i)]];
            if (acc > top_p) {
                for (int j = i + 1; j < n; j++)
                    probs[idx[static_cast<size_t>(j)]] = 0.0f;
                break;
            }
        }
        // Re-normalize
        sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0.0f)
            for (int i = 0; i < n; i++) probs[static_cast<size_t>(i)] /= sum;
    }

    // Categorical sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[static_cast<size_t>(i)];
        if (r <= cdf) return i;
    }
    return n - 1;
}

// ============================================================================
// Text Generator
// ============================================================================

class Generator {
public:
    Generator(const Model& model, GenConfig gc, unsigned seed = 42)
        : model_(model), gc_(gc), rng_(seed) {}

    // callback(token_id, decoded_text) is called for each generated token.
    void generate(const std::string& prompt,
                  std::function<void(int, const std::string&)> callback) {
        // Encode prompt — split on spaces for simplicity
        std::vector<int> tokens;
        const auto& tok = model_.tokenizer();
        tokens.push_back(tok.bos());

        // Simple space-tokenisation of prompt
        std::istringstream iss(prompt);
        std::string word;
        while (iss >> word) {
            int id = tok.lookup(word);
            if (id >= 0) tokens.push_back(id);
        }

        for (int step = 0; step < gc_.max_tokens; step++) {
            std::vector<float> logits = model_.forward(tokens);
            int next = sample_top_p(logits.data(), static_cast<int>(logits.size()),
                                    gc_.temp, gc_.top_p, rng_);
            if (next == tok.eos()) break;
            tokens.push_back(next);
            std::string piece = tok.decode(next);
            callback(next, piece);
        }
    }

private:
    const Model& model_;
    GenConfig    gc_;
    std::mt19937 rng_;
};

} // namespace slm
