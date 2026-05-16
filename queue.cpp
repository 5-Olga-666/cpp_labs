// test.cpp
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <algorithm>

template <typename T> class threadsafe_stack {
  mutable std::mutex mtx;
  std::condition_variable consumers, producers;
  std::vector<T> data;
  int readIdx = 0;
  int writeIdx = -1;
  bool done = false;

public:
  threadsafe_stack() {}
  threadsafe_stack(int size) : data(size) {}
  void push(T value);
  void wait_and_pop(T &dst);
  void wake_and_done();

private:
  bool full() const noexcept {
    return writeIdx >= static_cast<int>(data.size()) - 1;
  }
  bool empty() const noexcept { return writeIdx < 0; }
  bool no_new_tasks() const noexcept { return done; }
};

template <typename T> void threadsafe_stack<T>::push(T value) {
  std::unique_lock<std::mutex> lck{mtx};
  producers.wait(lck, [this]() { return !full() || no_new_tasks(); });
  if (no_new_tasks())
    return;
  ++writeIdx;
  data[(writeIdx + readIdx) % data.size()] = value;
  consumers.notify_one();
}

template <typename T> void threadsafe_stack<T>::wait_and_pop(T &dst) {
  std::unique_lock<std::mutex> lck{mtx};
  consumers.wait(lck, [this]() { return !empty() || no_new_tasks(); });
  if (empty())
    return;
  dst = data[readIdx % data.size()];
  readIdx = (readIdx + 1) % data.size();
  --writeIdx;
  producers.notify_one();
}

template <typename T> void threadsafe_stack<T>::wake_and_done() {
  done = true;
  consumers.notify_all();
  producers.notify_all();
}

TEST(ThreadsafeStackTest, DataLossUnderMultipleOperations) {
  threadsafe_stack<int> stack(10);
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;
  std::vector<int> received;
  std::mutex received_mtx;
  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};
  
  const int NUM_PRODUCERS = 5;
  const int NUM_CONSUMERS = 5;
  const int ITEMS_PER_PRODUCER = 100;
  
  for (int i = 0; i < NUM_PRODUCERS; ++i) {
    producers.emplace_back([&, i]() {
      for (int j = 0; j < ITEMS_PER_PRODUCER; ++j) {
        int value = i * ITEMS_PER_PRODUCER + j;
        stack.push(value);
        produced_count++;
      }
    });
  }
  
  // Консюмеры
  for (int i = 0; i < NUM_CONSUMERS; ++i) {
    consumers.emplace_back([&]() {
      for (;;) {
        int value;
        stack.wait_and_pop(value);
        if (value == -1 && stack.empty()) break; // Сигнал завершения
        {
          std::lock_guard<std::mutex> lock(received_mtx);
          received.push_back(value);
        }
        consumed_count++;
      }
    });
  }
  
  for (auto& p : producers) p.join();
  
  // Ждем пока все данные не будут обработаны
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stack.wake_and_done();
  
  for (auto& c : consumers) c.join();
  
  // ОШИБКА 1: Некоторые данные могут быть потеряны из-за неправильной логики full/empty
  EXPECT_EQ(produced_count, consumed_count) 
      << "Lost data: produced=" << produced_count << ", consumed=" << consumed_count;
  EXPECT_EQ(produced_count, received.size());
}

// Тест 2: Проверка на дублирование данных
TEST(ThreadsafeStackTest, DataDuplication) {
  threadsafe_stack<int> stack(5);
  std::vector<int> results;
  std::mutex results_mtx;
  
  // Продюсер
  std::thread producer([&]() {
    for (int i = 0; i < 10; ++i) {
      stack.push(i);
    }
    stack.wake_and_done();
  });
  
  // Несколько консюмеров
  std::vector<std::thread> consumers;
  for (int i = 0; i < 3; ++i) {
    consumers.emplace_back([&]() {
      int value;
      stack.wait_and_pop(value);
      {
        std::lock_guard<std::mutex> lock(results_mtx);
        results.push_back(value);
      }
    });
  }
  
  producer.join();
  for (auto& c : consumers) c.join();
  
  // ОШИБКА 2: Возможны дубликаты из-за неправильного управления индексами
  std::sort(results.begin(), results.end());
  auto duplicates = std::adjacent_find(results.begin(), results.end());
  EXPECT_EQ(duplicates, results.end()) 
      << "Found duplicate values in results";
}

// Тест 3: Проверка на гонку данных при одновременных операциях
TEST(ThreadsafeStackTest, RaceConditionUnderLoad) {
  threadsafe_stack<int> stack(20);
  std::atomic<bool> stop{false};
  std::atomic<int> total_pushed{0};
  std::atomic<int> total_popped{0};
  
  std::thread producer([&]() {
    for (int i = 0; i < 1000; ++i) {
      stack.push(i);
      total_pushed++;
      std::this_thread::yield();
    }
    stop = true;
  });
  
  std::thread consumer([&]() {
    while (!stop || !stack.empty()) {
      int value;
      stack.wait_and_pop(value);
      total_popped++;
    }
  });
  
  producer.join();
  stack.wake_and_done();
  consumer.join();
  
  // ОШИБКА 3: Из-за гонки данных total_pushed != total_popped
  EXPECT_EQ(total_pushed, total_popped) 
      << "Race condition detected: pushed=" << total_pushed 
      << ", popped=" << total_popped;
}

// Тест 4: Проверка на дедлок
TEST(ThreadsafeStackTest, DeadlockTest) {
  threadsafe_stack<int> stack(3);
  std::atomic<bool> producer_stuck{true};
  std::atomic<bool> consumer_stuck{true};
  
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  
  // Заполняем очередь до full
  for (int i = 0; i < 3; ++i) {
    stack.push(i);
  }
  
  std::thread producer([&]() {
    // Эта попытка push должна заблокироваться, т.к. очередь полна
    stack.push(100);
    producer_stuck = false;
  });
  
  std::thread consumer([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int value;
    stack.wait_and_pop(value);
    consumer_stuck = false;
  });
  
  // Ожидаем с таймаутом
  while (std::chrono::steady_clock::now() < deadline) {
    if (!producer_stuck && !consumer_stuck) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  bool deadlock_detected = (producer_stuck || consumer_stuck);
  stack.wake_and_done();
  producer.join();
  consumer.join();
  
  // ОШИБКА 4: Возможен дедлок из-за неправильной сигнализации condition_variable
  EXPECT_FALSE(deadlock_detected) 
      << "Deadlock detected! Producer stuck=" << producer_stuck 
      << ", Consumer stuck=" << consumer_stuck;
}

// Тест 5: Проверка на корректность порядка элементов (FIFO)
TEST(ThreadsafeStackTest, FIFOOrderTest) {
  threadsafe_stack<int> stack(10);
  std::vector<int> pushed;
  std::vector<int> popped;
  std::mutex popped_mtx;
  
  // Продюсеры в порядке
  std::thread producer([&]() {
    for (int i = 0; i < 50; ++i) {
      pushed.push_back(i);
      stack.push(i);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    stack.wake_and_done();
  });
  
  std::thread consumer([&]() {
    int value;
    while (stack.wait_and_pop(value), !stack.empty() || !stack.no_new_tasks()) {
      {
        std::lock_guard<std::mutex> lock(popped_mtx);
        popped.push_back(value);
      }
    }
  });
  
  producer.join();
  consumer.join();
  
  if (pushed.size() == popped.size()) {
    for (size_t i = 0; i < pushed.size(); ++i) {
      EXPECT_EQ(pushed[i], popped[i]) 
          << "FIFO order violated at index " << i;
    }
  }
}

// fixed_queue.h
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <optional>

template <typename T> class threadsafe_queue {
private:
  mutable std::mutex mtx;
  std::condition_variable data_cond;
  std::vector<T> data;
  size_t read_pos = 0;
  size_t write_pos = 0;
  size_t count = 0;
  bool done = false;

public:
  threadsafe_queue() = default;
  
  explicit threadsafe_queue(size_t capacity) : data(capacity) {}
  
  void push(T value) {
    std::unique_lock<std::mutex> lock(mtx);
    
    // Ожидаем место в очереди или сигнал завершения
    data_cond.wait(lock, [this] { 
      return count < data.size() || done; 
    });
    
    if (done) return;
    
    data[write_pos] = std::move(value);
    write_pos = (write_pos + 1) % data.size();
    ++count;
    
    data_cond.notify_one();
  }
  
  bool wait_and_pop(T& dst) {
    std::unique_lock<std::mutex> lock(mtx);
    
    // Ожидаем данные или сигнал завершения
    data_cond.wait(lock, [this] { 
      return count > 0 || done; 
    });
    
    if (count == 0) return false;
    
    dst = std::move(data[read_pos]);
    read_pos = (read_pos + 1) % data.size();
    --count;
    
    data_cond.notify_one();
    return true;
  }
  
  std::optional<T> wait_and_pop() {
    std::unique_lock<std::mutex> lock(mtx);
    
    data_cond.wait(lock, [this] { 
      return count > 0 || done; 
    });
    
    if (count == 0) return std::nullopt;
    
    T value = std::move(data[read_pos]);
    read_pos = (read_pos + 1) % data.size();
    --count;
    
    data_cond.notify_one();
    return value;
  }
  
  bool try_pop(T& dst) {
    std::lock_guard<std::mutex> lock(mtx);
    if (count == 0) return false;
    
    dst = std::move(data[read_pos]);
    read_pos = (read_pos + 1) % data.size();
    --count;
    
    return true;
  }
  
  void wake_and_done() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      done = true;
    }
    data_cond.notify_all();
  }
  
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return count == 0;
  }
  
  bool is_done() const {
    std::lock_guard<std::mutex> lock(mtx);
    return done;
  }
  
  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return count;
  }
  
  size_t capacity() const {
    std::lock_guard<std::mutex> lock(mtx);
    return data.size();
  }
};

// Тесты для исправленной версии
TEST(FixedThreadsafeQueueTest, NoDataLoss) {
  threadsafe_queue<int> queue(100);
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;
  std::atomic<int> produced{0};
  std::atomic<int> consumed{0};
  
  const int NUM_PROD = 5;
  const int NUM_CONS = 5;
  const int ITEMS_PER_PROD = 200;
  
  for (int i = 0; i < NUM_PROD; ++i) {
    producers.emplace_back([&, i]() {
      for (int j = 0; j < ITEMS_PER_PROD; ++j) {
        queue.push(i * ITEMS_PER_PROD + j);
        produced++;
      }
    });
  }
  
  for (int i = 0; i < NUM_CONS; ++i) {
    consumers.emplace_back([&]() {
      int value;
      while (queue.wait_and_pop(value)) {
        consumed++;
      }
    });
  }
  
  for (auto& p : producers) p.join();
  
  // Даем время на обработку
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  queue.wake_and_done();
  
  for (auto& c : consumers) c.join();
  
  EXPECT_EQ(produced, consumed);
  EXPECT_EQ(produced, NUM_PROD * ITEMS_PER_PROD);
}

TEST(FixedThreadsafeQueueTest, FIFOOrderPreserved) {
  threadsafe_queue<int> queue(50);
  std::vector<int> pushed;
  std::vector<int> popped;
  
  for (int i = 0; i < 100; ++i) {
    pushed.push_back(i);
    queue.push(i);
  }
  
  int value;
  while (queue.wait_and_pop(value)) {
    popped.push_back(value);
  }
  
  ASSERT_EQ(pushed.size(), popped.size());
  for (size_t i = 0; i < pushed.size(); ++i) {
    EXPECT_EQ(pushed[i], popped[i]);
  }
}

TEST(FixedThreadsafeQueueTest, CircularBufferWorks) {
  threadsafe_queue<int> queue(10);
  
  // Заполняем
  for (int i = 0; i < 10; ++i) {
    queue.push(i);
  }
  
  // Извлекаем несколько
  int val;
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(queue.wait_and_pop(val));
    EXPECT_EQ(val, i);
  }
  
  // Добавляем новые - они должны идти в освободившиеся слоты
  for (int i = 10; i < 15; ++i) {
    queue.push(i);
  }
  
  // Проверяем порядок
  for (int i = 5; i < 15; ++i) {
    EXPECT_TRUE(queue.wait_and_pop(val));
    EXPECT_EQ(val, i);
  }
  
  EXPECT_TRUE(queue.empty());
  queue.wake_and_done();
}