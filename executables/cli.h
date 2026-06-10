#ifndef QEW_CLI_H
#define QEW_CLI_H

#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Tiny `--key value` / `--key=value` command-line parser shared by the
// simulation executables, so parameter sweeps need no rebuild. It is
// deliberately dependency-free (standard headers only, no Kokkos/qew types) so
// it can be unit-tested standalone and included from any main.cpp.
//
// Conventions:
//   * Every option takes a value; the only flags are --help / -h.
//   * Values may be negative numbers -- a value starts with a single '-',
//     whereas an option key is marked by the leading '--'.
//   * Defaults are supplied at each read site (the executables keep their
//     former module-constant values as defaults, so no-argument runs are
//     byte-for-byte unchanged).
//   * After reading every known option, call require_all_used() to reject
//     typos / unknown keys.
namespace qew::cli {

class Args {
public:
    Args(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string tok = argv[i];
            if (tok == "--help" || tok == "-h") {
                help = true;
                continue;
            }
            if (tok.rfind("--", 0) != 0)
                throw std::runtime_error("unexpected argument '" + tok +
                                         "' (options start with --)");
            std::string key = tok.substr(2);
            std::string val;
            const auto eq = key.find('=');
            if (eq != std::string::npos) {
                val = key.substr(eq + 1);
                key = key.substr(0, eq);
            } else if (i + 1 < argc) {
                val = argv[++i];  // next token is the value (may be negative)
            } else {
                throw std::runtime_error("option --" + key + " needs a value");
            }
            if (key.empty())
                throw std::runtime_error("empty option name in '" + tok + "'");
            kv_[key] = val;
        }
    }

    bool help = false;

    bool has(const std::string &k) const { return kv_.count(k) != 0; }

    std::string str(const std::string &k, const std::string &def) {
        const auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        used_.insert(k);
        return it->second;
    }
    double dbl(const std::string &k, double def) {
        const auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        used_.insert(k);
        return parse_one<double>(k, it->second);
    }
    int integer(const std::string &k, int def) {
        const auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        used_.insert(k);
        return parse_one<int>(k, it->second);
    }
    std::uint64_t u64(const std::string &k, std::uint64_t def) {
        const auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        used_.insert(k);
        return parse_one<std::uint64_t>(k, it->second);
    }
    // Comma-separated list of doubles, e.g. --pinning_lengths 0.1,1,10.
    std::vector<double> dbl_list(const std::string &k,
                                 const std::vector<double> &def) {
        const auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        used_.insert(k);
        std::vector<double> out;
        std::stringstream ss(it->second);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            out.push_back(parse_one<double>(k, item));
        }
        if (out.empty())
            throw std::runtime_error("option --" + k + ": empty list '" +
                                     it->second + "'");
        return out;
    }

    // Throw if any supplied option was never read (typo / unknown key).
    void require_all_used() const {
        std::string unknown;
        for (const auto &kv : kv_)
            if (!used_.count(kv.first))
                unknown += (unknown.empty() ? "" : ", ") + ("--" + kv.first);
        if (!unknown.empty())
            throw std::runtime_error("unknown option(s): " + unknown);
    }

private:
    template <class T>
    static T parse_one(const std::string &k, const std::string &v) {
        std::istringstream ss(v);
        T out{};
        ss >> out;
        if (ss.fail() || ss.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("option --" + k + ": cannot parse '" + v +
                                     "' as a number");
        return out;
    }

    std::map<std::string, std::string> kv_;
    std::set<std::string> used_;
};

}  // namespace qew::cli

#endif  // QEW_CLI_H
