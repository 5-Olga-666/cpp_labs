#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <future>
#include <algorithm>

template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    // Конструкторы
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t n, size_t m) : rows(n), cols(m) {
        data.resize(rows, std::vector<T>(cols, T()));
    }
    
    Matrix(size_t n, size_t m, const T& value) : rows(n), cols(m) {
        data.resize(rows, std::vector<T>(cols, value));
    }
    
    // Конструктор копирования
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}
    
    // Доступ к элементам
    T& operator()(size_t i, size_t j) {
        return data[i][j];
    }
    
    const T& operator()(size_t i, size_t j) const {
        return data[i][j];
    }
    
    // Получение размеров
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    // Оператор транспонирования (~)
    Matrix<T> operator~() const {
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    // Оператор умножения (*)
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        Matrix<T> result(rows, other.cols, T());
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    // Умножение с предварительным транспонированием (строка на строку)
    Matrix<T> multiplyWithTranspose(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        Matrix<T> transposed = ~other;
        Matrix<T> result(rows, other.cols, T());
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * transposed(j, k);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    // Заполнение случайными значениями
    void fillRandom(T min = 0, T max = 100) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }
    
    // Вывод матрицы
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Параллельное умножение матриц (задание 3)
    Matrix<T> parallelMultiply(const Matrix<T>& other, size_t num_threads = std::thread::hardware_concurrency()) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        Matrix<T> result(rows, other.cols, T());
        
        // Разделяем строки результата между потоками
        std::vector<std::thread> threads;
        size_t rows_per_thread = rows / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start_row = t * rows_per_thread;
            size_t end_row = (t == num_threads - 1) ? rows : start_row + rows_per_thread;
            
            threads.emplace_back([this, &other, &result, start_row, end_row]() {
                for (size_t i = start_row; i < end_row; ++i) {
                    for (size_t j = 0; j < other.cols; ++j) {
                        T sum = T();
                        for (size_t k = 0; k < cols; ++k) {
                            sum += data[i][k] * other(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return result;
    }
};