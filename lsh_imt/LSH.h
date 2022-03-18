#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <map>

class LSH
{
	private:
		// Members
		int counter;
		int rnd;
		const int MAX_SIZE;
		const int K;
		const int L;
		const int THREADS;
		int* rset;
		std::map<int, int> label_map;
		std::vector<std::unordered_map<int, std::vector<int>>> tables;

		// Functions
		void add(const int, const int, const int);
		void add_multi(const int*, const int, const int);

		void retrieve(int*, const int, const int);
		void retrieve(std::unordered_set<int> ,const int, const int);
		void retrieve_multi(int*, const int*, const int, const int);
        void retrieve_batch(const int*, long*, const int, const size_t, const int);

	public:
		LSH(int, int, int, int);
		~LSH();
		void insert(const int*, const int);
		void insert_multi(const int*, const int);
		void insert_multi_label(const int*, const int*, const int N);
		std::unordered_set<int> query(const int*);
		std::vector<int> query_multi(const int*, const int);
		std::vector<int> query_multi_label(const int*, const int N, int c);
		void query_batch(const int*, long*, const int, const size_t);
		void clear();
		void count();
};
