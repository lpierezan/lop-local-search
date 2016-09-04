# include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <random>

using namespace std;

// ============= Utility =====================================
#define D_INF (DBL_MAX)
#define I32_INF (1<<31)
auto rand_gen = std::default_random_engine(0);
void reset_rand(int seed) {
	rand_gen = std::default_random_engine(seed);
}

string pii_string(pair<int, int> pii) {
	return ("(" + to_string(pii.first) + " , " + to_string(pii.second) + ")");
}
//============================================================

// =============== Time/Clock Measurement ====================
using  ms = chrono::milliseconds;
using  ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;
#define to_time(x) (chrono::duration_cast<ms>(x).count())
// ===========================================================

class lazy_treap {

	/*
		Sliding Boxes

		Treap (with min-heap property) data structure such that each element has the <key,value> format.
		The elements are sorted by key.

		Supports in O(logN):
		ins(<Key k, Value v>) : adds element <k,v>
		del(Key k) : remove element with Key = k
		find(Key k) : returns element with Key = k
		max_value() : returns the max value in the tree.
		sum_value(Key k´, Value delta_value) : For each element <k,v> such that k > k´ sums delta to the value, v += delta_value.
		sum_key(Key k´, delta_key) : for each element <k,v> such that k > k´, sums delta_key to his key, k += delta_key.*
		
		*(1) in sum_key an element it not updated twice, the key comparsion is made with prior update keys.
		*(2) in sum_key, if delta_key < 0, its expected that min{k | k > k´} + delta_key > min{k | k <= k´} 
		, otherwise binary search tree invariant could be invalidated.

		Some important invariants:
		- left subtree keys < root key <= right subtree keys
		- min-heap property over priority
		- the node max_value does not include lazy data.

	*/

	const static int treap_inf = I32_INF;
	
public:
	class node {
	public:
		//element variables
		int key, value;
		//subtree variables
		int max_value, lazy_value, lazy_key, count;
		//treap variables
		int priority;
		node *l,*r;

		node(int _key = 0, int _value = 0) :key(_key), value(_value), max_value(_value), lazy_value(0), lazy_key(0),
			count(1), l(NULL), r(NULL)
		{
			priority = (rand_gen()) % treap_inf;
		}

		// ==== Node Normalization Methods =====
		void update() {
			this->count = 1; 
			this->max_value = value + lazy_value;
			
			if (this->l != NULL) {
				this->count += this->l->count;
				this->max_value = max(this->max_value, this->l->max_value + this->l->lazy_value);
			}

			if (this->r != NULL) {
				this->count += this->r->count;
				this->max_value = max(this->max_value, this->r->max_value + this->r->lazy_value);
			}
		}

		void lazy_propagation() {
			if (this->lazy_key == 0 && this->lazy_value == 0) return;

			this->key += lazy_key;
			this->value += lazy_value;
			this->max_value += lazy_value;

			if (this->l != NULL) {
				this->l->lazy_key += lazy_key;
				this->l->lazy_value += lazy_value;
			}
			if (this->r != NULL) {
				this->r->lazy_key += lazy_key;
				this->r->lazy_value += lazy_value;
			}

			this->lazy_key = this->lazy_value = 0;
		}
		// ===========================

	};

	node *root;

	lazy_treap() {
		root = NULL;
	}

private:
	
	// ==== Rotations ========
	void rr(node *&p) {
		//assume p and p->l != NULL and p not lazy
		node *q = p->l;
		q->lazy_propagation();
		p->l = q->r;
		q->r = p;
		p->update(); q->update();
		p = q;
	}

	void lr(node *&p) {
		//assume p and p->r != NULL and p not lazy
		node *q = p->r;
		q->lazy_propagation();
		p->r = q->l;
		q->l = p;
		p->update(); q->update();
		p = q;
	}
	// =======================

	void ins(node *&p, pair<int,int> &kv_pair) {
		if (p == NULL) {
			p = new node(kv_pair.first,kv_pair.second);			
		}
		else {
			p->lazy_propagation();
			if (kv_pair.first < p->key) {
				ins(p->l, kv_pair);
				if (p->l->priority < p->priority) rr(p);
			}
			else
			{
				ins(p->r, kv_pair);
				if (p->r->priority < p->priority) lr(p);
			}
		}
		p->update();
	}

	void del(node *&p, int key) {
		if (p == NULL) return;
		p->lazy_propagation();

		if (p->key == key) 
			del(p);
		else
			if (key < p->key) del(p->l, key);
			else del(p->r, key);

		if (p != NULL) p->update();
	}

	void del(node *&p) {
		//assume p is not lazy
		if (p->l == NULL && p->r == NULL) { delete p; p = NULL; return; }

		if (p->r == NULL || (p->l != NULL && p->l->priority <= p->r->priority)) {
			//left node goes up
			rr(p); del(p->r);
		}
		else {
			//right node goes up
			lr(p); del(p->l);
		}

		p->update();
	}

	bool find(node *&p, int key, pair<int,int> &ret) {
		if (p == NULL) return false;
		p->lazy_propagation();

		if (key == p->key) {
			ret.first = p->key; ret.second = p->value;
			return true;
		}

		if (key < p->key) return find(p->l, key, ret); else return find(p->r, key, ret);
	}

	void rfs(node *&p, int k, pair<int, int> &ret) {
		//assumes k >= 1 and k <= size of rooted tree in p
		p->lazy_propagation();

		int lsize = p->l != NULL ? p->l->count : 0;
		int rsize = p->r != NULL ? p->r->count : 0;

		if (k <= lsize) return rfs(p->l, k, ret);
		else if (k == lsize + 1) {
			ret.first = p->key; ret.second = p->value;
			return;
		}
		else return rfs(p->r, k - p->l->count - 1, ret);
	}

	void sum_value(node *&p, int key, int d_value) {
		if (p == NULL) return;
		p->lazy_propagation();

		if (p->key <= key) {
			sum_value(p->r, key, d_value);
		}
		else { //p->key > key
			p->value += d_value;
			if (p->r != NULL) p->r->lazy_value += d_value;
			sum_value(p->l, key, d_value);
		}
		p->update();
	}

	void sum_key(node *&p, int key, int d_key) {
		if (p == NULL) return;
		p->lazy_propagation();

		if (p->key <= key) {
			sum_value(p->r, key, d_key);
		}
		else { //p->key > key
			p->key += d_key;
			if (p->r != NULL) p->r->lazy_key += d_key;
			sum_value(p->l, key, d_key);
		}
		p->update();
	}

	void print(node *&p, string &ret) {
		if (p == NULL) return;
		p->lazy_propagation();

		print(p->l, ret);
		ret = ret + pii_string( make_pair(p->key,p->value) ) + " ";
		print(p->r, ret);
	}
public:
	void ins(pair<int,int> kv_pair) { ins(root, kv_pair); }
	void del(int key) { del(root, key); }
	bool find(int key, pair<int, int> &ret) { return find(root, key, ret); }
	bool rfs(int k, pair<int, int> &ret) { 
		if (k >= 1 && k <= root->count) { rfs(root, k, ret); return true; }
		else return false; 
	}
	int max_value() {
		if (root == NULL) return (-I32_INF);
		root->lazy_propagation();
		return root->max_value;
	}
	void sum_value(int key, int d_value) {
		return sum_value(root, key, d_value);
	}
	void sum_key(int key, int d_key) {
		return sum_key(root, key, d_key);
	}

	string to_string() {
		string ret = "";
		print(root,ret);
		return ret;
	}

};

class array_imp {
	vector<pair<int, int>> v;

public:
	array_imp() {};

	void ins(pair<int, int> kv_pair) { v.push_back(kv_pair); }
	void del(int key) {
		for (int i = 0; i < v.size(); i++) {
			if (v[i].first == key) { v.erase(v.begin() + i); break; }
		}
	}

	bool find(int key, pair<int, int> &ret) { 
		int i;
		for (i = 0; i < v.size(); i++) {
			if (v[i].first == key) { ret = v[i]; break; }
		}
		return (i < v.size());
	}
	
	int max_value() {
		int ret = (-I32_INF);
		for (int i = 0; i < v.size(); i++)
		{
			ret = max(ret, v[i].second);
		}
		return ret;
	}

	void sum_value(int key, int d_value) {
		for (int i = 0; i < v.size(); i++)
		{
			if (v[i].first > key) { v[i].second += d_value; }
		}
	}

	void sum_key(int key, int d_key) {
		for (int i = 0; i < v.size(); i++)
		{
			if (v[i].first > key) { v[i].first += d_key; }
		}
	}

	string to_string() {
		string ret = "";
		sort(v.begin(), v.end());
		for (int i = 0; i < v.size(); i++) {
			ret = ret + pii_string(v[i]) + " ";
		}
	}
};



int main() {
	
	char * op = new char[32];
	lazy_treap tree;
	array_imp array;

	int n = 10,i, w_max = 100;
	for (i = 1; i <= n; i++) {
		int w = 1 + rand_gen() % w_max;
		tree.ins(make_pair(i, w));
		array.ins(make_pair(i, w));
	}

	int K, V, D, P1, P2;
	bool allways_print_array = true;
	pair<int, int> ret;
	string tret, aret;

	while (true) {
		cin >> op;
		tret = ""; aret = "";

		//1 - insert I K V
		if (op == "I") {
			cin >> K >> V;
			tree.ins(make_pair(K, V));
			array.ins(make_pair(K, V));
		}
		//2 - find F K
		if (op == "F") {
			cin >> K;

			if (tree.find(K, ret)) {
				tret = pii_string(ret);
				
			}

			if (array.find(K, ret)) {
				aret = pii_string(ret);
			}
		}

		//3 - remove D K
		if (op == "D") {
			cin >> K;

		}

		//4 - max_value MV
		if (op == "MV") {

		}
		
		//5 - sum_value SV K DV
		if (op == "SV") {

		}

		//6 - sum_key SK K DK
		if (op == "SK") {

		}
		//7 - print P
		if (op == "P") {

		}

		//8 - LOP_Insert LI P1 P2
		if (op == "LI") {

		}

		cout << tret << endl << aret << endl;

		if (op != "P" && allways_print_array) {
			cout << array.to_string() << endl;
		}
	}

	return 0;
}