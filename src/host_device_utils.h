#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "xrt/xrt_device.h"

namespace host_device {

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline bool is_supported_card_name(const std::string& name) {
    const std::string n = lower(name);
    return n.find("u250") != std::string::npos || n.find("u280") != std::string::npos;
}

inline std::string board_label(const std::string& name) {
    const std::string n = lower(name);
    if (n.find("u280") != std::string::npos) return "U280";
    if (n.find("u250") != std::string::npos) return "U250";
    return name;
}

inline std::string target_card_from_env() {
    const char* target = std::getenv("TARGET_CARD");
    if (!target || !*target) target = std::getenv("BOARD");
    return target ? lower(target) : std::string{};
}

inline int auto_select_supported_device() {
    const std::string target = target_card_from_env();
    int first_supported = -1;

    for (int i = 0; ; ++i) {
        try {
            xrt::device probe(i);
            const std::string dname = probe.get_info<xrt::info::device::name>();
            const std::string lname = lower(dname);
            std::cout << "[HOST] device[" << i << "]: " << dname << "\n";

            if (!is_supported_card_name(dname)) continue;
            if (!target.empty()) {
                if (lname.find(target) != std::string::npos) return i;
            } else if (first_supported < 0) {
                first_supported = i;
            }
        } catch (...) {
            break;
        }
    }

    if (!target.empty()) {
        throw std::runtime_error("No supported device matching target card '" + target + "'");
    }
    if (first_supported < 0) {
        throw std::runtime_error("No supported U250/U280 device found");
    }
    return first_supported;
}

inline int parse_device_arg(const std::string& arg) {
    const std::string a = lower(arg);
    if (a == "auto" || a == "-1") return auto_select_supported_device();
    return std::stoi(arg);
}

inline void require_supported_device(const std::string& selected_name) {
    if (!is_supported_card_name(selected_name)) {
        throw std::runtime_error("Selected device is not U250/U280: " + selected_name);
    }
}

}  // namespace host_device
