#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <chrono>
#include <optional>
#include <algorithm>
#include <atomic>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::mutex cout_mutex;

template<typename... Args>
void safe_print(Args&&... args) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    (std::cout << ... << std::forward<Args>(args)) << std::flush;
}

struct Task {
    int row_index;
    std::vector<uint8_t> pixels;
};

template <typename T>
class BlockingQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    bool shutdown_ = false;

public:
    explicit BlockingQueue(size_t max_size) : max_size_(max_size) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return queue_.size() < max_size_ || shutdown_;
        });
        if (shutdown_) return false;
        queue_.push(std::move(item));
        cv_.notify_all();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return !queue_.empty() || shutdown_;
        });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cv_.notify_all();
        return item;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

class SynchronizedImageCollector {
private:
    std::vector<std::vector<uint8_t>> image_data_;
    std::mutex mutex_;
    std::atomic<int> completed_rows_{0};
    int total_rows_;

public:
    explicit SynchronizedImageCollector(int total_rows)
        : image_data_(total_rows), total_rows_(total_rows) {}

    void add_result(int row_index, std::vector<uint8_t> processed_row) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            image_data_[row_index] = std::move(processed_row);
        }
        ++completed_rows_;
    }

    std::vector<std::vector<uint8_t>> get_final_image() {
        return image_data_;
    }

    int completed() const {
        return completed_rows_.load();
    }
};

void consumer_worker(BlockingQueue<Task>& task_queue, SynchronizedImageCollector& collector) {
    while (true) {
        auto task_opt = task_queue.pop();
        if (!task_opt.has_value()) {
            break;
        }
        Task& task = task_opt.value();
        std::vector<uint8_t> inverted;
        inverted.reserve(task.pixels.size());
        for (uint8_t pixel : task.pixels) {
            inverted.push_back(static_cast<uint8_t>(255 - pixel));
        }
        collector.add_result(task.row_index, std::move(inverted));
    }
}

int main() {
    int num_consumers = 0;
    while (num_consumers <= 0) {
        std::cout << "Введите количество потоков: ";
        if (!(std::cin >> num_consumers) || num_consumers <= 0) {
            std::cout << "Неверно, введите целое положительное число.\n";
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            num_consumers = 0;
        }
    }

    std::cin.ignore();
    std::string input_filename;
    std::cout << "Введите имя файла: ";
    std::getline(std::cin, input_filename);

    input_filename.erase(
        std::remove(input_filename.begin(), input_filename.end(), '\"'),
        input_filename.end()
    );

    size_t dot_pos = input_filename.find_last_of('.');
    std::string output_filename = (dot_pos == std::string::npos)
        ? input_filename + "_inverted.jpg"
        : input_filename.substr(0, dot_pos) + "_inverted" + input_filename.substr(dot_pos);

    int width, height, channels;
    unsigned char* img_data = stbi_load(input_filename.c_str(), &width, &height, &channels, 3);
    if (!img_data) {
        std::cout << "Не удалось загрузить файл: " << input_filename << "\n";
        return 1;
    }

    const size_t QUEUE_MAX_SIZE = static_cast<size_t>(num_consumers) * 4;
    BlockingQueue<Task> task_queue(QUEUE_MAX_SIZE);
    SynchronizedImageCollector collector(height);

    std::vector<std::thread> consumers;
    consumers.reserve(num_consumers);

    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back(consumer_worker, std::ref(task_queue), std::ref(collector));
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; ++y) {
        Task task;
        task.row_index = y;
        task.pixels.reserve(width * 3);
        const uint8_t* row_start = img_data + y * width * 3;
        task.pixels.assign(row_start, row_start + width * 3);
        if (!task_queue.push(std::move(task))) {
            break;
        }
    }

    stbi_image_free(img_data);
    task_queue.shutdown();

    for (auto& t : consumers) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Обработано строк: " << collector.completed() << "/" << height << "\n";
    std::cout << "Время: " << elapsed.count() << " сек.\n";

    auto final_matrix = collector.get_final_image();
    std::vector<uint8_t> flat_image;
    flat_image.reserve(static_cast<size_t>(width) * height * 3);
    for (const auto& row : final_matrix) {
        flat_image.insert(flat_image.end(), row.begin(), row.end());
    }

    int save_result = 0;
    if (output_filename.size() >= 4 &&
        output_filename.substr(output_filename.size() - 4) == ".png") {
        save_result = stbi_write_png(output_filename.c_str(), width, height, 3,
            flat_image.data(), width * 3);
    } else {
        save_result = stbi_write_jpg(output_filename.c_str(), width, height, 3,
            flat_image.data(), 95);
    }

    if (save_result == 0) {
        std::cout << "Ошибка сохранения\n";
        return 1;
    }

    std::cout << "Сохранено: " << output_filename << "\n";
    return 0;
}