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
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t n, size_t m) : rows(n), cols(m) {
        data.resize(rows, std::vector<T>(cols, T()));
    }
    
    Matrix(size_t n, size_t m, const T& value) : rows(n), cols(m) {
        data.resize(rows, std::vector<T>(cols, value));
    }
    
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}
    
    T& operator()(size_t i, size_t j) {
        return data[i][j];
    }
    
    const T& operator()(size_t i, size_t j) const {
        return data[i][j];
    }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    Matrix<T> operator~() const {
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Несоответствие размеров матрицы для умножения");
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
    
    Matrix<T> multiplyWithTranspose(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Несоответствие размеров матрицы для умножения");
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
    
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    Matrix<T> parallelMultiply(const Matrix<T>& other, size_t num_threads = std::thread::hardware_concurrency()) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Несоответствие размеров матрицы для умножения");
        }
        
        Matrix<T> result(rows, other.cols, T());
        
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

template<typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

void benchmarkMatrixMultiplication() {
    const size_t N = 200;
    
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    
    A.fillRandom(0, 10);
    B.fillRandom(0, 10);
    
    std::cout << "Тест на умножение матриц (size: " << N << "x" << N << ")\n";
    
    double time_direct = measureTime([&]() {
        Matrix<double> C = A * B;
    });
    std::cout << "Прямое умножение (row*col): " << time_direct << " ms\n";
    
    double time_transpose = measureTime([&]() {
        Matrix<double> C = A.multiplyWithTranspose(B);
    });
    std::cout << "Умножение с транспонированием (row*row): " << time_transpose << " ms\n";
    
    double time_parallel = measureTime([&]() {
        Matrix<double> C = A.parallelMultiply(B);
    });
    std::cout << "Параллельное умножение: " << time_parallel << " ms\n";
    
    std::cout << "\nSpeedup (direct/parallel): " << (time_direct / time_parallel) << "x\n";
}

class PingPongGame {
private:
    std::mutex mtx;
    std::condition_variable cv;
    bool is_ping_turn = true;
    bool game_running = true;
    int position = 0;
    const int MAX_POSITION = 40;
    
    void clearScreen() {
        std::cout << "\033[2J\033[1;1H";
    }
    
    void drawProgress(int pos, bool is_ping_moving) {
        std::cout << "T1 ";
        
        for (int i = 0; i < MAX_POSITION; ++i) {
            if (is_ping_moving && i == pos) {
                std::cout << "o";
            } else if (!is_ping_moving && i == MAX_POSITION - 1 - pos) {
                std::cout << "o";
            } else {
                std::cout << "-";
            }
        }
        
        std::cout << " T2";
        
        if (!is_ping_moving) {
            std::cout << " (T2 notified T1)";
        }
        std::cout << std::endl;
    }
    
public:
    void ping_thread() {
        while (game_running && position < MAX_POSITION) {
            std::unique_lock<std::mutex> lock(mtx);
            
            cv.wait(lock, [this] { return is_ping_turn || !game_running; });
            
            if (!game_running) break;

            for (int i = 0; i <= MAX_POSITION; ++i) {
                clearScreen();
                position = i;
                drawProgress(position, true);
                std::cout << "T1 --------|-------- T2 (PING sending...)" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            std::cout << "T1 sent 'ping' to T2" << std::endl;
            is_ping_turn = false;
            cv.notify_one();
        }
    }
    
    void pong_thread() {
        while (game_running && position < MAX_POSITION) {
            std::unique_lock<std::mutex> lock(mtx);

            cv.wait(lock, [this] { return !is_ping_turn || !game_running; });
            
            if (!game_running) break;

            for (int i = MAX_POSITION; i >= 0; --i) {
                clearScreen();
                drawProgress(i, false);
                std::cout << "T1 --------|-------- T2 (PONG returning...)" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            std::cout << "T2 sent 'pong' to T1" << std::endl;
            is_ping_turn = true;
            cv.notify_one();
        }
    }
    
    void run(int rounds = 3) {
        std::cout << "Starting Ping-Pong game!\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        for (int round = 0; round < rounds; ++round) {
            position = 0;
            std::thread t1(&PingPongGame::ping_thread, this);
            std::thread t2(&PingPongGame::pong_thread, this);
            
            t1.join();
            t2.join();
            
            if (round < rounds - 1) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                clearScreen();
                std::cout << "\nRound " << (round + 1) << " completed! Next round...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        game_running = false;
        clearScreen();
        std::cout << "Игра окончена!\n";
    }
};

class SimplePingPong {
private:
    std::mutex mtx;
    std::condition_variable cv;
    bool is_ping_turn = true;
    int ping_count = 0;
    int pong_count = 0;
    const int MAX_COUNT = 5;
    
public:
    void ping() {
        for (int i = 0; i < MAX_COUNT; ++i) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return is_ping_turn; });
            
            std::cout << "Пинг " << ++ping_count << std::endl;
            is_ping_turn = false;
            cv.notify_one();
        }
    }
    
    void pong() {
        for (int i = 0; i < MAX_COUNT; ++i) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return !is_ping_turn; });
            
            std::cout << "Понг " << ++pong_count << std::endl;
            is_ping_turn = true;
            cv.notify_one();
        }
    }
};

int main() {
    std::cout << "=== Задания 1 и 2: сравнительный анализ матричных операций ===\n\n";
    benchmarkMatrixMultiplication();
    
    std::cout << "\n\n=== Задание 3. Параллельное умножение матриц ===\n";
    const size_t N = 500;
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    A.fillRandom();
    B.fillRandom();
    
    double parallel_time = measureTime([&]() {
        Matrix<double> C = A.parallelMultiply(B);
    });
    std::cout << "Время параллельного умножения: " << parallel_time << " ms\n";
    
    std::cout << "\n\n=== Задание 4: игра в пинг-понг ===\n";
    std::cout << "Выберите режим:\n";
    std::cout << "1. Простой пинг-понг (без визуализации)\n";
    std::cout << "2. Анимированный пинг-понг с визуализацией\n";
    
    int choice;
    std::cin >> choice;
    
    if (choice == 1) {
        SimplePingPong game;
        std::thread t1(&SimplePingPong::ping, &game);
        std::thread t2(&SimplePingPong::pong, &game);
        t1.join();
        t2.join();
    } else if (choice == 2) {
        PingPongGame game;
        game.run(3);
    }
    
    return 0;
}
