#include <iterator>
#include <future>
#include <vector>
#include <functional>
#include <thread>

template<typename Iterator, typename UnaryFunc, typename BinaryFunc>
auto map_reduce_async(Iterator begin, Iterator end, UnaryFunc f1, BinaryFunc f2, size_t num_threads) {
    using ResultType = decltype(f1(*begin));
    size_t length = std::distance(begin, end);
    size_t chunk_size = length / num_threads;
    size_t remainder = length % num_threads;

    std::vector<std::future<ResultType>> futures;

    Iterator chunk_start = begin;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        Iterator chunk_end = std::next(chunk_start, current_chunk_size);

        futures.push_back(std::async(std::launch::async,
            [chunk_start, chunk_end, f1, f2]() {
                auto it = chunk_start;
                ResultType res = f1(*it);
                ++it;
                for (; it != chunk_end; ++it) {
                    res = f2(res, f1(*it));
                }
                return res;
            }
        ));

        chunk_start = chunk_end;
    }

    ResultType final_result = futures[0].get();
    for (size_t i = 1; i < futures.size(); ++i) {
        final_result = f2(final_result, futures[i].get());
    }
    return final_result;
}

template<typename Iterator, typename UnaryFunc, typename BinaryFunc>
auto map_reduce_thread(Iterator begin, Iterator end, UnaryFunc f1, BinaryFunc f2, size_t num_threads) {
    using ResultType = decltype(f1(*begin));
    size_t length = std::distance(begin, end);
    size_t chunk_size = length / num_threads;
    size_t remainder = length % num_threads;

    std::vector<std::thread> threads;
    std::vector<ResultType> results(num_threads);

    Iterator chunk_start = begin;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        Iterator chunk_end = std::next(chunk_start, current_chunk_size);

        threads.emplace_back([chunk_start, chunk_end, f1, f2, &results, i]() {
            auto it = chunk_start;
            ResultType res = f1(*it);
            ++it;
            for (; it != chunk_end; ++it) {
                res = f2(res, f1(*it));
            }
            results[i] = res;
        });

        chunk_start = chunk_end;
    }

    for (auto& t : threads) {
        t.join();
    }

    ResultType final_result = results[0];
    for (size_t i = 1; i < results.size(); ++i) {
        final_result = f2(final_result, results[i]);
    }
    return final_result;
}

int main()
{
    return 0;
}