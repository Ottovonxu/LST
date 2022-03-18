#include "LSH.h"
#include <string.h>
#include <stddef.h>
#include <thread>
#include <iostream>

LSH::LSH(int MAX_SIZE_, int K_, int L_, int T_) : counter(0), rnd(0), MAX_SIZE(MAX_SIZE_), K(K_), L(L_), THREADS(T_)
{

	rset = new int[MAX_SIZE];
	for(int idx = 0; idx < L; ++idx)
	{
		std::unordered_map<int, std::vector<int>> table;
		tables.emplace_back(std::move(table));
	}
	std::map<int, int> label_map;
}

LSH::~LSH()
{
	delete [] rset;
}

// Insert N elements into LSH hash tables
void LSH::insert_multi(const int* fp, const int N)
{
	std::vector<std::thread> thread_list;
	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { add_multi(fp, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}
}

void LSH::count()
{

	for (int l=0; l<L ; l++){
		std::cout<<"table: "<<l;
		for (int b=0; b< (1<<K); b++){
			std::cout<<b <<":" << tables[l][b].size() <<" ";
		}
		std::cout<< std::endl;
	}
}

void LSH::insert_multi_label(const int* fp, const int* labels, const int N)
{
	for (int tdx = 0; tdx < N; ++tdx){
		label_map[tdx] = labels[tdx];
	}

	std::vector<std::thread> thread_list;
	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { add_multi(fp, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}
}

// Insert a single element into LSH hash tables
void LSH::insert(const int* fp, const int item_id)
{
	for(int idx = 0; idx < L; ++idx)
	{
		add(fp[idx], idx, item_id);
	}
}

// Parallel Insert: Each thread handle an independent set of tables - (N x L)
void LSH::add_multi(const int* fp, const int N, const int tdx)
{
	// For each example
	for(int idx = 0; idx < N; ++idx)
	{
		// For each table
		for(int jdx = tdx; jdx < L; jdx+=THREADS)
		{
			add(fp[idx * L + jdx], jdx, idx);
		}
	}
}

// Insert a item; Check if bucket exists first
void LSH::add(const int key, const int idx, const int item_id)
{
	std::unordered_map<int, std::vector<int>>& table = tables[idx];
	if(table.find(key) == table.end())
	{
		std::vector<int> value;
		table.emplace(key, std::move(value));
	}
	table[key].push_back(item_id);
}

std::unordered_set<int> LSH::query(const int* fp)
{
	std::unordered_set<int> result;
	for(int idx = 0; idx < L; ++idx)
	{
		int key = fp[idx];
		retrieve(result, idx, key);
	}
	return result;
}

std::vector<int> LSH::query_multi(const int* fp, const int N)
{
	std::vector<std::thread> thread_list;
	memset(rset, 0, sizeof(int) * MAX_SIZE);

	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { retrieve_multi(rset, fp, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}

	std::vector<int> result;
	for(int idx = 0; idx < MAX_SIZE; ++idx)
	{
		if(rset[idx]>2)
		{
			result.push_back(idx);
		}
	}
	++rnd;
	counter += result.size();
	return result;
}

std::vector<int> LSH::query_multi_label(const int* fp, const int N, int c)
{
	std::vector<std::thread> thread_list;
	memset(rset, 0, sizeof(int) * MAX_SIZE);

	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { retrieve_multi(rset, fp, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}

	std::vector<int> result;
	for(int idx = 0; idx < MAX_SIZE; ++idx)
	{
		if(rset[idx])
		{
			if (label_map[idx]!=c){
				result.push_back(idx);
			}
		}
	}
	++rnd;
	counter += result.size();
	return result;
}

void LSH::query_batch(const int* fp, long* result, const int N, const size_t SIZE)
{
	++rnd;
	std::vector<std::thread> thread_list;
	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { retrieve_batch(fp, result, N, SIZE, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}
}

// (L x N) matrix - for each table - N examples
void LSH::retrieve_multi(int* rset, const int* fp, const int N, const int offset)
{
	std::unordered_set<int> visited;

	// For each table
	for(int idx = offset; idx < L; idx+=THREADS)
	{
		// For each example
		for(int jdx = 0; jdx < N; ++jdx)
		{
			const int index = idx * N + jdx;
			const int key = fp[index];
			if(visited.find(key) == visited.end())
			{
				retrieve(rset, idx, key);
				visited.emplace(key);
			}
		}
	}
}

void LSH::retrieve(int* rset, const int table_idx, const int bucket_idx)
{
	std::unordered_map<int, std::vector<int>>& table = tables[table_idx];
	if(table.find(bucket_idx) != table.end())
	{
		const std::vector<int>& bucket = table[bucket_idx];
		for(int idx : bucket)
		{
			rset[idx] += 1;
		}
	}
}

void LSH::retrieve(std::unordered_set<int> result, const int table_idx, const int bucket_idx)
{
	std::unordered_map<int, std::vector<int>>& table = tables[table_idx];
	if(table.find(bucket_idx) != table.end())
	{
		const std::vector<int>& bucket = table[bucket_idx];
		result.insert(bucket.begin(), bucket.end());
	}
}


// (L x N) matrix - for each table - N examples
void LSH::retrieve_batch(const int* fp, long* result, const int N, const size_t SIZE, const int tdx)
{
	std::unordered_set<int> rset;

	// For each example
	for(int idx = tdx; idx < N; idx+=THREADS)
	{
		rset.clear();

		// For each table
		for(int jdx = 0; jdx < L; ++jdx)
		{
			const int key = fp[idx * L + jdx];
			retrieve(rset, jdx, key);
		}

		{
			size_t jdx = 0;
			for(auto iter = rset.begin(); jdx < SIZE && iter != rset.end(); ++jdx, ++iter)
			{
				const int index = idx * SIZE + jdx;
				result[index] = *iter;
			}
		}
	}

	if(!tdx)
	{
		counter += rset.size();
	}
}


void LSH::clear()
{
	int avg_size = counter / std::max(rnd, 1);
	printf("Clear: %d\n", avg_size);
	counter = 0;
	rnd = 0;

	for(int idx = 0; idx < L; ++idx)
	{
		tables[idx].clear();
	}

}
