#pragma once
// ============================================================================
// slm_engine.hpp — Lightweight C++17 Inference Engine for Small Language Models
// Single-file / header-only. Zero external dependencies beyond OS APIs + STL.
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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#define SLM_OMP_PRAGMA(value) _Pragma(#value)
#define SLM_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define SLM_OMP_PARALLEL_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#define SLM_OMP_SIMD _Pragma("omp simd")
#define SLM_OMP_SIMD_REDUCE(variable) SLM_OMP_PRAGMA(omp simd reduction(+:variable))
#else
#define SLM_OMP_PRAGMA(value)
#define SLM_OMP_PARALLEL_FOR
#define SLM_OMP_PARALLEL_FOR_COLLAPSE_2
#define SLM_OMP_SIMD
#define SLM_OMP_SIMD_REDUCE(variable)
#endif

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
// MappedFile — Cross-platform read-only file mapping wrapper
// ============================================================================

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& o) noexcept
        : data_(o.data_), size_(o.size_)
#if defined(_WIN32) || defined(_WIN64)
        , file_(o.file_), mapping_(o.mapping_)
#else
        , fd_(o.fd_)
#endif
    {
        o.data_ = nullptr;
        o.size_ = 0;
#if defined(_WIN32) || defined(_WIN64)
        o.file_ = INVALID_HANDLE_VALUE;
        o.mapping_ = nullptr;
#else
        o.fd_ = -1;
#endif
    }

    bool open(const char* path) {
#if defined(_WIN32) || defined(_WIN64)
        file_ = ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "[SLM ERROR] Cannot open '%s' (Windows error %lu)\n",
                    path, static_cast<unsigned long>(::GetLastError()));
            return false;
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file_, &file_size) || file_size.QuadPart <= 0) {
            fprintf(stderr, "[SLM ERROR] Cannot determine size of '%s'\n", path);
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);
        if (static_cast<LONGLONG>(size_) != file_size.QuadPart) {
            fprintf(stderr, "[SLM ERROR] File '%s' is too large for this process\n", path);
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            size_ = 0;
            return false;
        }

        mapping_ = ::CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            fprintf(stderr, "[SLM ERROR] CreateFileMapping failed for '%s' (Windows error %lu)\n",
                    path, static_cast<unsigned long>(::GetLastError()));
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            size_ = 0;
            return false;
        }

        void* p = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!p) {
            fprintf(stderr, "[SLM ERROR] MapViewOfFile failed for '%s' (Windows error %lu)\n",
                    path, static_cast<unsigned long>(::GetLastError()));
            ::CloseHandle(mapping_);
            ::CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            size_ = 0;
            return false;
        }
        data_ = static_cast<const uint8_t*>(p);
        return true;
#else
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
#endif
    }

    void close() {
#if defined(_WIN32) || defined(_WIN64)
        if (data_) ::UnmapViewOfFile(data_);
        if (mapping_) ::CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
#else
        if (data_ && data_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
        }
        if (fd_ >= 0) ::close(fd_);
#endif
        data_ = nullptr;
        size_ = 0;
#if defined(_WIN32) || defined(_WIN64)
        mapping_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        fd_ = -1;
#endif
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
#if defined(_WIN32) || defined(_WIN64)
    HANDLE          file_ = INVALID_HANDLE_VALUE;
    HANDLE          mapping_ = nullptr;
#else
    int            fd_   = -1;
#endif
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
    float temp       = 0.7f;
    float top_p      = 0.9f;
};

struct KVCache {
    int n_layers = 0;
    int max_seq_len = 0;
    int kv_width = 0;
    std::vector<float> keys;
    std::vector<float> values;

    void initialize(const ModelConfig& config) {
        n_layers = config.n_layers;
        max_seq_len = config.max_seq_len;
        kv_width = config.n_kv_heads * config.d_head();
        size_t layer_size = static_cast<size_t>(max_seq_len) * kv_width;
        keys.assign(static_cast<size_t>(n_layers) * layer_size, 0.0f);
        values.assign(static_cast<size_t>(n_layers) * layer_size, 0.0f);
    }

    float* key(int layer, int pos) {
        size_t index = (static_cast<size_t>(layer) * max_seq_len + pos) * kv_width;
        return keys.data() + index;
    }

    float* value(int layer, int pos) {
        size_t index = (static_cast<size_t>(layer) * max_seq_len + pos) * kv_width;
        return values.data() + index;
    }
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
    void load(const JsonValue& vocab_obj, const JsonValue& metadata = JsonValue()) {
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

        unk_id_ = metadata["unk_id"].integer(lookup(metadata["unk_token"].as_string()));
        if (!valid_id(unk_id_)) unk_id_ = lookup("<unk>");
        if (!valid_id(unk_id_)) unk_id_ = lookup("<|unk|>");
        if (!valid_id(unk_id_)) unk_id_ = id2tok_.empty() ? 0 : 0;

        bos_id_ = metadata["bos_id"].integer(-1);
        if (!valid_id(bos_id_)) bos_id_ = lookup(metadata["bos_token"].as_string());
        if (!valid_id(bos_id_)) bos_id_ = lookup("<bos>");
        if (!valid_id(bos_id_)) bos_id_ = lookup("<s>");
        if (!valid_id(bos_id_)) bos_id_ = lookup("<|endoftext|>");
        if (!valid_id(bos_id_)) bos_id_ = unk_id_;

        eos_id_ = metadata["eos_id"].integer(-1);
        if (!valid_id(eos_id_)) eos_id_ = lookup(metadata["eos_token"].as_string());
        if (!valid_id(eos_id_)) eos_id_ = lookup("<eos>");
        if (!valid_id(eos_id_)) eos_id_ = lookup("</s>");
        if (!valid_id(eos_id_)) eos_id_ = lookup("<|endoftext|>");
        if (!valid_id(eos_id_)) eos_id_ = -1;
    }

    int lookup(const std::string& tok) const {
        auto it = tok2id_.find(tok);
        return it != tok2id_.end() ? it->second : -1;
    }

    int encode(const std::string& tok) const {
        int id = lookup(tok);
        return valid_id(id) ? id : unk_id_;
    }

    int encode_word(const std::string& word, bool leading_space,
                    bool leading_newline = false) const {
        if (word.empty()) return unk_id_;

        int id = lookup(word);
        if (valid_id(id)) return id;

        if (leading_newline) {
            id = lookup(std::string("Ċ") + word);
            if (valid_id(id)) return id;
        }
        if (leading_space) {
            id = lookup(std::string("Ġ") + word);
            if (valid_id(id)) return id;
        }

        // Accept already byte-level encoded input while also handling a raw
        // prompt that contains these marker characters.
        std::string sanitized = strip_byte_marker(word);
        id = lookup(sanitized);
        return valid_id(id) ? id : unk_id_;
    }

    std::string decode(int id) const {
        if (id < 0 || id >= static_cast<int>(id2tok_.size())) return "<unk>";
        return id2tok_[static_cast<size_t>(id)];
    }

    int bos() const { return bos_id_; }
    int eos() const { return eos_id_; }
    int unk() const { return unk_id_; }
    bool valid_id(int id) const {
        return id >= 0 && id < static_cast<int>(id2tok_.size());
    }

private:
    static std::string strip_byte_marker(const std::string& token) {
        if (token.compare(0, 2, "Ġ") == 0 || token.compare(0, 2, "Ċ") == 0)
            return token.substr(2);
        return token;
    }

    std::unordered_map<std::string, int> tok2id_;
    std::vector<std::string>             id2tok_;
    int unk_id_ = 0;
    int bos_id_ = 0;
    int eos_id_ = -1;
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
        // New layout: [uint32 LE header length][padded JSON][tensor data].
        // Keep reading the older [JSON][NUL][tensor data] layout as well.
        JsonParser parser;
        JsonValue root;
        bool length_prefixed = false;
        uint32_t header_length = 0;
        if (file_.size() >= sizeof(header_length)) {
            std::memcpy(&header_length, file_.data(), sizeof(header_length));
            length_prefixed = header_length > 0 &&
                              header_length % alignof(float) == 0 &&
                              static_cast<size_t>(header_length) + sizeof(header_length) < file_.size() &&
                              file_.data()[sizeof(header_length)] == '{';
        }

        size_t json_start = 0;
        size_t json_len = 0;
        std::string json_storage;
        if (length_prefixed) {
            json_start = sizeof(header_length);
            json_len = header_length;
            json_storage.assign(reinterpret_cast<const char*>(file_.data() + json_start),
                                json_len);
            json_storage.push_back('\0');
            root = parser.parse(json_storage.c_str());
        } else {
            while (json_len < file_.size() && file_.data()[json_len] != 0)
                ++json_len;
            if (json_len == 0 || json_len >= file_.size())
                SLM_DIE_FMT("Model: cannot find JSON header in '%s'", path);
            root = parser.parse(file_.cstr());
        }

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
        tokenizer_.load(root["vocab"], root);

        // ---- Tensor metadata ----
        const JsonValue& tensors_json = root["tensors"];
        if (!tensors_json.is_array()) SLM_DIE("Model: missing 'tensors' array");

        tensor_data_offset_ = static_cast<size_t>(root["tensor_data_offset"].integer());
        if (tensor_data_offset_ == 0 || tensor_data_offset_ >= file_.size())
            SLM_DIE("Model: invalid 'tensor_data_offset'");
        if (length_prefixed && tensor_data_offset_ != sizeof(header_length) + header_length)
            SLM_DIE("Model: tensor_data_offset does not match header length");

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
            if (ti.dtype != "f32")
                SLM_DIE_FMT("Model: tensor '%s' has dtype '%s' (only f32 supported)",
                             ti.name.c_str(), ti.dtype.c_str());
            if (ti.offset % alignof(float) != 0 || ti.byte_size() % alignof(float) != 0)
                SLM_DIE_FMT("Model: tensor '%s' is not float-aligned", ti.name.c_str());
            if (ti.offset > file_.size() - tensor_data_offset_ ||
                ti.byte_size() > file_.size() - tensor_data_offset_ - ti.offset)
                SLM_DIE_FMT("Model: tensor '%s' exceeds mapped file", ti.name.c_str());
            name2tensor_[ti.name] = tensors_.size();
            tensors_.push_back(std::move(ti));
        }

        // ---- Resolve weight pointers ----
        const uint8_t* base_bytes = file_.offset(tensor_data_offset_);
        if (reinterpret_cast<uintptr_t>(base_bytes) % alignof(float) != 0)
            SLM_DIE("Model: tensor data is not float-aligned");
        const float* base = reinterpret_cast<const float*>(base_bytes);

        auto ptr = [&](const std::string& name) -> const float* {
            auto it = name2tensor_.find(name);
            if (it == name2tensor_.end()) {
                SLM_DIE_FMT("Model: missing tensor '%s'", name.c_str());
            }
            const TensorInfo& ti = tensors_[it->second];
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

    // Incremental forward pass for one token. The cache contains K/V entries
    // for positions [0, pos], so the caller can generate without recomputing
    // the entire prefix on every step.
    std::vector<float> forward_cached(int token, int pos, KVCache& cache) const;

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
    SLM_OMP_PARALLEL_FOR
    for (int i = 0; i < d_out; i++) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        const float* row = W + static_cast<size_t>(i) * d_in;
        SLM_OMP_SIMD_REDUCE(sum)
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
    SLM_OMP_SIMD_REDUCE(ss)
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / n + eps);
    SLM_OMP_SIMD
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
    SLM_OMP_PARALLEL_FOR
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
    if (tokens.empty()) SLM_DIE("Model::forward: token sequence is empty");
    if (tokens.size() > static_cast<size_t>(config_.max_seq_len))
        SLM_DIE("Model::forward: token sequence exceeds max_seq_len");
    for (int token : tokens) {
        if (token < 0 || token >= config_.vocab_size)
            SLM_DIE_FMT("Model::forward: token %d out of range", token);
    }

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
    SLM_OMP_PARALLEL_FOR
    for (int t = 0; t < T; t++) {
        std::memcpy(&x[static_cast<size_t>(t) * D],
                     tok_emb_ + tokens[static_cast<size_t>(t)] * D,
                     D * sizeof(float));
    }

    // 2. Transformer layers
    const int kv_rep = H / KH;   // GQA repeat factor

    for (int l = 0; l < config_.n_layers; l++) {
        // --- Attention layer norm ---
        SLM_OMP_PARALLEL_FOR
        for (int t = 0; t < T; t++) {
            rms_norm(&x[static_cast<size_t>(t) * D],
                     layer_attn_norm_[l],
                     &h[static_cast<size_t>(t) * D], D, eps);
        }

        // --- Q, K, V projections ---
        SLM_OMP_PARALLEL_FOR
        for (int t = 0; t < T; t++) {
            const float* ht = &h[static_cast<size_t>(t) * D];
            linear(ht, layer_wq_[l], &q[static_cast<size_t>(t) * H * DH], D, H * DH);
            linear(ht, layer_wk_[l], &k[static_cast<size_t>(t) * KH * DH], D, KH * DH);
            linear(ht, layer_wv_[l], &v[static_cast<size_t>(t) * KH * DH], D, KH * DH);
        }

        // --- RoPE ---
        SLM_OMP_PARALLEL_FOR
        for (int t = 0; t < T; t++) {
            rope_apply(&q[static_cast<size_t>(t) * H * DH],
                       &k[static_cast<size_t>(t) * KH * DH],
                       H, KH, DH, t, thr);
        }

        // --- Scaled dot-product attention with causal mask ---
        const float scale = 1.0f / std::sqrt(static_cast<float>(DH));
        SLM_OMP_PARALLEL_FOR_COLLAPSE_2
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
                        SLM_OMP_SIMD_REDUCE(dot)
                        for (int i = 0; i < DH; i++) dot += qh[i] * kh[i];
                        scores[static_cast<size_t>(t2)] = dot * scale;
                    }
                }
                softmax(scores.data(), T);

                // Weighted sum of values
                float* out_h = &attn_out[static_cast<size_t>(t) * D + h * DH];
                for (int i = 0; i < DH; i++) {
                    float s = 0.0f;
                    SLM_OMP_SIMD_REDUCE(s)
                    for (int t2 = 0; t2 <= t; t2++) {
                        s += scores[static_cast<size_t>(t2)] *
                             v[static_cast<size_t>(t2) * KH * DH + kv_h * DH + i];
                    }
                    out_h[i] = s;
                }
            }
        }

        // --- Output projection ---
        SLM_OMP_PARALLEL_FOR
        for (int t = 0; t < T; t++) {
            linear(&attn_out[static_cast<size_t>(t) * D],
                   layer_wo_[l],
                   &proj[static_cast<size_t>(t) * D], D, D);
        }

        // --- Residual ---
        const size_t residual_size = static_cast<size_t>(T) * D;
        SLM_OMP_PARALLEL_FOR
        for (size_t i = 0; i < residual_size; i++)
            x[i] += proj[i];

        // --- FFN layer norm ---
        for (int t = 0; t < T; t++) {
            rms_norm(&x[static_cast<size_t>(t) * D],
                     layer_ffn_norm_[l],
                     &h[static_cast<size_t>(t) * D], D, eps);
        }

        // --- FFN: gate_proj(W1) -> SiLU, up_proj(W3), SwiGLU, down_proj(W2) ---
        SLM_OMP_PARALLEL_FOR
        for (int t = 0; t < T; t++) {
            size_t bt = static_cast<size_t>(t);
            const float* ht = &h[bt * D];
            linear(ht, layer_w1_[l], &ff_gate[bt * FF], D, FF);
            linear(ht, layer_w3_[l], &ff_up[bt * FF],   D, FF);
            swiglu(&ff_gate[bt * FF], &ff_up[bt * FF], &ff_down[bt * FF], FF);
            linear(&ff_down[bt * FF], layer_w2_[l], &proj[bt * D], FF, D);
        }

        // --- Residual ---
        SLM_OMP_PARALLEL_FOR
        for (size_t i = 0; i < residual_size; i++)
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
        SLM_OMP_SIMD_REDUCE(dot)
        for (int j = 0; j < D; j++) dot += last_hidden[j] * row[j];
        logits[static_cast<size_t>(vi)] = dot;
    }

    return logits;
}

inline std::vector<float> Model::forward_cached(int token, int pos, KVCache& cache) const {
    const int D = config_.d_model;
    const int H = config_.n_heads;
    const int KH = config_.n_kv_heads;
    const int DH = config_.d_head();
    const int FF = config_.d_ff;
    const int V = config_.vocab_size;

    if (token < 0 || token >= V)
        SLM_DIE_FMT("Model::forward_cached: token %d out of range", token);
    if (pos < 0 || pos >= config_.max_seq_len)
        SLM_DIE_FMT("Model::forward_cached: position %d exceeds max_seq_len %d",
                     pos, config_.max_seq_len);
    if (cache.n_layers != config_.n_layers ||
        cache.max_seq_len != config_.max_seq_len ||
        cache.kv_width != KH * DH)
        SLM_DIE("Model::forward_cached: incompatible KV cache");

    std::vector<float> x(D);
    std::memcpy(x.data(), tok_emb_ + static_cast<size_t>(token) * D,
                static_cast<size_t>(D) * sizeof(float));
    std::vector<float> h(D);
    std::vector<float> q(static_cast<size_t>(H) * DH);
    std::vector<float> k(static_cast<size_t>(KH) * DH);
    std::vector<float> v(static_cast<size_t>(KH) * DH);
    std::vector<float> attn_out(D);
    std::vector<float> proj(D);
    std::vector<float> ff_gate(FF);
    std::vector<float> ff_up(FF);
    std::vector<float> ff_down(FF);

    const int kv_rep = H / KH;
    const float eps = config_.norm_eps;
    const float theta = config_.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(DH));

    for (int layer = 0; layer < config_.n_layers; layer++) {
        rms_norm(x.data(), layer_attn_norm_[layer], h.data(), D, eps);
        linear(h.data(), layer_wq_[layer], q.data(), D, H * DH);
        linear(h.data(), layer_wk_[layer], k.data(), D, KH * DH);
        linear(h.data(), layer_wv_[layer], v.data(), D, KH * DH);
        rope_apply(q.data(), k.data(), H, KH, DH, pos, theta);

        std::memcpy(cache.key(layer, pos), k.data(),
                    static_cast<size_t>(KH * DH) * sizeof(float));
        std::memcpy(cache.value(layer, pos), v.data(),
                    static_cast<size_t>(KH * DH) * sizeof(float));

        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        SLM_OMP_PARALLEL_FOR
        for (int head = 0; head < H; head++) {
            const int kv_head = head / kv_rep;
            const float* q_head = q.data() + head * DH;
            std::vector<float> scores(static_cast<size_t>(pos) + 1);
            for (int past = 0; past <= pos; past++) {
                const float* k_head = cache.key(layer, past) + kv_head * DH;
                float dot = 0.0f;
                SLM_OMP_SIMD_REDUCE(dot)
                for (int i = 0; i < DH; i++) dot += q_head[i] * k_head[i];
                scores[static_cast<size_t>(past)] = dot * scale;
            }
            softmax(scores.data(), pos + 1);

            float* out_head = attn_out.data() + head * DH;
            for (int i = 0; i < DH; i++) {
                float sum = 0.0f;
                SLM_OMP_SIMD_REDUCE(sum)
                for (int past = 0; past <= pos; past++) {
                    sum += scores[static_cast<size_t>(past)] *
                           (cache.value(layer, past) + kv_head * DH)[i];
                }
                out_head[i] = sum;
            }
        }

        linear(attn_out.data(), layer_wo_[layer], proj.data(), D, D);
        for (int i = 0; i < D; i++) x[static_cast<size_t>(i)] += proj[static_cast<size_t>(i)];

        rms_norm(x.data(), layer_ffn_norm_[layer], h.data(), D, eps);
        linear(h.data(), layer_w1_[layer], ff_gate.data(), D, FF);
        linear(h.data(), layer_w3_[layer], ff_up.data(), D, FF);
        swiglu(ff_gate.data(), ff_up.data(), ff_down.data(), FF);
        linear(ff_down.data(), layer_w2_[layer], proj.data(), FF, D);
        for (int i = 0; i < D; i++) x[static_cast<size_t>(i)] += proj[static_cast<size_t>(i)];
    }

    std::vector<float> last_hidden(D);
    rms_norm(x.data(), final_norm_, last_hidden.data(), D, config_.norm_eps);
    std::vector<float> logits(V);
    for (int vocab = 0; vocab < V; vocab++) {
        const float* row = tok_emb_ + static_cast<size_t>(vocab) * D;
        float dot = 0.0f;
        SLM_OMP_SIMD_REDUCE(dot)
        for (int i = 0; i < D; i++) dot += last_hidden[static_cast<size_t>(i)] * row[i];
        logits[static_cast<size_t>(vocab)] = dot;
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
    if (n <= 0) SLM_DIE("Sampling requires at least one logit");
    bool has_nonzero = false;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(logits[i])) SLM_DIE("Sampling received non-finite logits");
        if (logits[i] != 0.0f) has_nonzero = true;
    }
    if (!has_nonzero) log_info("Warning: all logits are zero; sampling uniformly");
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
        const auto& tok = model_.tokenizer();
        KVCache cache;
        cache.initialize(model_.config());

        std::vector<int> prompt_tokens;
        int bos_id = tok.bos();
        if (!tok.valid_id(bos_id) || bos_id >= model_.config().vocab_size)
            bos_id = tok.unk();
        if (!tok.valid_id(bos_id) || bos_id >= model_.config().vocab_size)
            SLM_DIE("Generator: tokenizer has no valid BOS/UNK token");
        prompt_tokens.push_back(bos_id);

        std::string word;
        bool leading_space = false;
        bool leading_newline = false;
        auto append_word = [&]() {
            if (word.empty()) return;
            int id = tok.encode_word(word, leading_space, leading_newline);
            if (id < 0 || id >= model_.config().vocab_size)
                id = tok.unk();
            if (id < 0 || id >= model_.config().vocab_size)
                SLM_DIE_FMT("Generator: invalid token id %d for prompt word '%s'",
                             id, word.c_str());
            prompt_tokens.push_back(id);
            word.clear();
            leading_space = false;
            leading_newline = false;
        };

        for (char c : prompt) {
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

        if (prompt_tokens.size() > static_cast<size_t>(model_.config().max_seq_len))
            SLM_DIE("Generator: prompt exceeds model max_seq_len");

        int pos = 0;
        std::vector<float> logits;
        for (int token : prompt_tokens)
            logits = model_.forward_cached(token, pos++, cache);

        for (int step = 0; step < gc_.max_tokens && pos < model_.config().max_seq_len; step++) {
            int next = sample_top_p(logits.data(), static_cast<int>(logits.size()),
                                    gc_.temp, gc_.top_p, rng_);
            if (next == tok.eos() || next == 0 || next == 2) break;
            std::string piece = tok.decode(next);
            if (piece == "<|im_end|>" || piece == "<|endoftext|>" || piece == "</s>" || piece == "<eos>") break;
            callback(next, piece);

            // Feed the sampled token back through the one-token cached path.
            if (step + 1 >= gc_.max_tokens || pos >= model_.config().max_seq_len)
                break;
            logits = model_.forward_cached(next, pos++, cache);
        }
    }

private:
    const Model& model_;
    GenConfig    gc_;
    std::mt19937 rng_;
};

} // namespace slm

#undef SLM_OMP_PARALLEL_FOR
#undef SLM_OMP_PARALLEL_FOR_COLLAPSE_2
#undef SLM_OMP_SIMD
#undef SLM_OMP_SIMD_REDUCE
#undef SLM_OMP_PRAGMA
