#include <iostream>
#include <vector>
#include <memory>
#include <optional>
#include <random>
#include <cassert>

template <typename Key, typename Compare = std::less<Key>>
class SkipList {
private:
    struct Node {
        Key key;
        std::vector<std::shared_ptr<Node>> next;
        Node(const Key& k, size_t level) : key(k), next(level + 1, nullptr) {}
    };

    using NodePtr = std::shared_ptr<Node>;

    Compare comp;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    double probability;
    size_t maxLevel;
    NodePtr head;
    size_t countElements;

    int randomLevel() {
        int lvl = 0;
        while (dist(rng) < probability && lvl < maxLevel - 1) {
            ++lvl;
        }
        return lvl;
    }

    NodePtr findNode(const Key& key) const {
        NodePtr curr = head;
        for (int lvl = maxLevel - 1; lvl >= 0; --lvl) {
            while (curr->next[lvl] && comp(curr->next[lvl]->key, key)) {
                curr = curr->next[lvl];
            }
        }
        return curr->next[0];
    }

    std::vector<NodePtr> getPredecessors(const Key& key) const {
        std::vector<NodePtr> pred(maxLevel);
        NodePtr curr = head;
        for (int lvl = maxLevel - 1; lvl >= 0; --lvl) {
            while (curr->next[lvl] && comp(curr->next[lvl]->key, key)) {
                curr = curr->next[lvl];
            }
            pred[lvl] = curr;
        }
        return pred;
    }

public:
    class Iterator {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Key;
        using difference_type = std::ptrdiff_t;
        using pointer = const Key*;
        using reference = const Key&;

        Iterator() = default;
        Iterator(NodePtr node) : current(node) {}

        reference operator*() const { return current->key; }
        pointer operator->() const { return &current->key; }

        Iterator& operator++() {
            if (current) current = current->next[0];
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        Iterator& operator--() {
            // Not implemented for simplicity, but would require a reverse traversal structure.
            // In a real skip list, we'd need a backward pointer or start from head.
            throw std::logic_error("--iterator not implemented without prev pointers");
        }

        Iterator operator--(int) {
            Iterator tmp = *this;
            --(*this);
            return tmp;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.current == b.current;
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return a.current != b.current;
        }

    private:
        NodePtr current;
    };

    using reverse_iterator = std::reverse_iterator<Iterator>;

    SkipList(double p = 0.5, size_t maxLvl = 32)
        : rng(std::random_device{}()), dist(0.0, 1.0),
          probability(p), maxLevel(maxLvl), countElements(0) {
        head = std::make_shared<Node>(Key(), maxLevel - 1);
    }

    template <typename InputIt>
    SkipList(InputIt first, InputIt last, double p = 0.5, size_t maxLvl = 32)
        : SkipList(p, maxLvl) {
        for (auto it = first; it != last; ++it) {
            insert(*it);
        }
    }

    SkipList(const SkipList& other)
        : comp(other.comp), rng(other.rng), dist(other.dist),
          probability(other.probability), maxLevel(other.maxLevel), countElements(0) {
        head = std::make_shared<Node>(Key(), maxLevel - 1);
        for (auto it = other.begin(); it != other.end(); ++it) {
            insert(*it);
        }
    }

    SkipList(SkipList&& other) noexcept
        : comp(std::move(other.comp)), rng(std::move(other.rng)), dist(other.dist),
          probability(other.probability), maxLevel(other.maxLevel),
          head(std::move(other.head)), countElements(other.countElements) {
        other.countElements = 0;
        other.head = std::make_shared<Node>(Key(), maxLevel - 1);
    }

    SkipList& operator=(const SkipList& other) {
        if (this != &other) {
            SkipList tmp(other);
            swap(tmp);
        }
        return *this;
    }

    SkipList& operator=(SkipList&& other) noexcept {
        if (this != &other) {
            swap(other);
            other.clear();
        }
        return *this;
    }

    void swap(SkipList& other) noexcept {
        std::swap(comp, other.comp);
        std::swap(rng, other.rng);
        std::swap(dist, other.dist);
        std::swap(probability, other.probability);
        std::swap(maxLevel, other.maxLevel);
        std::swap(head, other.head);
        std::swap(countElements, other.countElements);
    }

    bool empty() const { return countElements == 0; }
    size_t size() const { return countElements; }

    void insert(const Key& key) {
        auto preds = getPredecessors(key);
        NodePtr curr = preds[0]->next[0];

        if (curr && curr->key == key) {
            // allow duplicates? Let's allow duplicates for count() method
            // For simplicity, we just insert duplicate as new node
        }

        int newLevel = randomLevel();
        NodePtr newNode = std::make_shared<Node>(key, newLevel);

        for (int lvl = 0; lvl <= newLevel; ++lvl) {
            newNode->next[lvl] = preds[lvl]->next[lvl];
            preds[lvl]->next[lvl] = newNode;
        }
        ++countElements;
    }

    template <typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (auto it = first; it != last; ++it) {
            insert(*it);
        }
    }

    Iterator find(const Key& key) const {
        NodePtr node = findNode(key);
        if (node && node->key == key) {
            return Iterator(node);
        }
        return end();
    }

    size_t count(const Key& key) const {
        size_t cnt = 0;
        NodePtr curr = findNode(key);
        while (curr && curr->key == key) {
            ++cnt;
            curr = curr->next[0];
        }
        return cnt;
    }

    Iterator lower_bound(const Key& key) const {
        NodePtr curr = head;
        for (int lvl = maxLevel - 1; lvl >= 0; --lvl) {
            while (curr->next[lvl] && comp(curr->next[lvl]->key, key)) {
                curr = curr->next[lvl];
            }
        }
        curr = curr->next[0];
        return Iterator(curr);
    }

    Iterator upper_bound(const Key& key) const {
        NodePtr curr = head;
        for (int lvl = maxLevel - 1; lvl >= 0; --lvl) {
            while (curr->next[lvl] && !comp(key, curr->next[lvl]->key)) {
                curr = curr->next[lvl];
            }
        }
        curr = curr->next[0];
        return Iterator(curr);
    }

    std::pair<Iterator, Iterator> equal_range(const Key& key) const {
        return {lower_bound(key), upper_bound(key)};
    }

    void erase(Iterator it) {
        if (it == end()) return;
        const Key& key = *it;
        auto preds = getPredecessors(key);
        NodePtr toDel = preds[0]->next[0];
        if (!toDel || toDel->key != key) return;

        for (size_t lvl = 0; lvl < toDel->next.size(); ++lvl) {
            preds[lvl]->next[lvl] = toDel->next[lvl];
        }
        --countElements;
    }

    void erase(Iterator first, Iterator last) {
        while (first != last) {
            erase(first++);
        }
    }

    void clear() {
        head = std::make_shared<Node>(Key(), maxLevel - 1);
        countElements = 0;
    }

    Iterator begin() const { return Iterator(head->next[0]); }
    Iterator end() const { return Iterator(nullptr); }
    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }
};