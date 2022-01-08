//
// Created by d3vyatk4 on 17.11.2021.
//

#ifndef BROBAND_ML_EXCEPTION_H
#define BROBAND_ML_EXCEPTION_H

#include <exception>
#include <string>

class NotImplemented : public std::exception {
private:
    std::string m_error;

public:
    NotImplemented(std::string error = "Function only has a carcass") : m_error(error) {}

    const char *what() const noexcept override {
        return m_error.c_str();
    }
};

#endif//BROBAND_ML_EXCEPTION_H
