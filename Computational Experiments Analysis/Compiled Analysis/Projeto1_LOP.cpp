// Projeto1_LOP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
# include <chrono>
#include <iostream>
#include <vector>
#include <queue>
#include <list>
#include <fstream>
#include <string>
#include <algorithm>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <random>
#include <functional>
#include <tuple>
#include <typeinfo>

using namespace std;
// ============= Utility =====================================
#define D_INF (DBL_MAX)
#define I32_INF (1<<31)
auto rand_gen = std::default_random_engine(0);
void reset_rand(int seed) {
	rand_gen = std::default_random_engine(seed);
}
long long int overflow_test = ((long long)1 << 62);
//clever online shuffle algorithm
int select_shuffle(vector<int> &v, int &max_idx) {
	auto pos = max_idx  > 0 ? rand_gen() % (max_idx+1) : 0;
	swap(v[pos], v[max_idx]);
	max_idx--;
	return v[max_idx + 1];
}

string pii_to_str(pair<int, int> p) { return "(" + to_string(p.first) + "," + to_string(p.second) + ")"; }
//============================================================


// =============== Time/Clock Measurement ====================
using  ms = chrono::milliseconds;
using get_time = chrono::steady_clock;
#define to_time(x) (chrono::duration_cast<ms>(x).count())

//global clock (TODO Improve design - make it a class)
auto global_time = get_time::now();
long long global_tl_sec;
bool active_time_limit = false;
void restart_global_clock(long long tl_sec, bool active_tl_test) { 
	global_time = get_time::now(); 
	global_tl_sec = tl_sec; 
	active_time_limit = active_tl_test;
}
bool tl_clock_check(long long &duration_ms) {
	duration_ms = to_time(get_time::now() - global_time);
	return active_time_limit && ((duration_ms/1000.0) >= global_tl_sec);
}
// ===========================================================

// ===================== Statistics ===================

template <typename Tnum>
class stats_stream {
	/* Collect stream of data and compute online statistics (good when not storing sequence)
	*/
	string name;
	bool do_sd = false, firstSample = true, save_seq = false,
		is_over_time = false;

	Tnum total = 0, minv, maxv;
	long double total_2;

	long nsample = 0;
	list<Tnum> seq;
	list<double> time_seq;

public:
	stats_stream(string _name) : name(_name) {}
	stats_stream(string _name, bool _do_sd, bool _save_seq = false, bool _over_time = false) 
		: name(_name), do_sd(_do_sd), save_seq(_save_seq), is_over_time(_over_time)
		{ };
	
	void add(Tnum x, double time_point = 0) {
		nsample++;
		firstSample ? (minv = x) : (minv = min(minv, x));
		firstSample ? (maxv = x) : (maxv = max(maxv, x));
		firstSample = false;
		total += x; if (total > overflow_test) cout << "Error Overflow risk" << endl, exit(1);
		if (do_sd) { 
			total_2 += x*x; 
			if (total_2 > overflow_test) { cout << "Error Overflow risk" << endl; exit(1); }
		}
		if (save_seq) {
			seq.push_back(x);
			if (is_over_time) time_seq.push_back(time_point);
		}
	}

	Tnum get_total() { return total; }
	double get_mean() {return nsample != 0 ? ((double)total / nsample) : 0;}
	pair<Tnum, Tnum> get_min_max() { return make_pair(minv, maxv); }
	double get_sd() {
		if (nsample == 0) return 0;
		//sqrt(E[X^2] - E[X]^2)
		auto var = (total_2 / nsample) - ((long double)total / nsample)*((double)total / nsample);
		return sqrtl(var);
	}
	Tnum get_max() { return maxv; };
	bool empty() { return nsample == 0; }

	void print(bool print_total = false) {
		cout << name << ": " << "min = " << minv << " max = " << maxv << " mean = " << get_mean();
		if (do_sd) cout << " sd = " << get_sd();
		if(print_total) cout << " total = " << total;
		cout << endl;
	}


};

class inst_stats {
	/* Represents statistics of a Instance. 
	TODO improve design (should be inside instance class maybe)
	*/
public:	
	
	void init(int _n) {
		n = _n;
		v_impact = vector<double>(n);
	}

	// time stats
	long clean_time = 0;
	// graph stats
	int n;
	int edge_number = 0, edge_number_clean = 0;	
	long n_scc = 0;
	stats_stream<long> scc_sizes = stats_stream<long>("SCC sizes");
	vector<double> v_impact;
	int ind_set_size = 0;

	void print() {
		cout << "E(G*)/E(G) = " << 100*((double)edge_number_clean/edge_number) << "%" << endl;		
		cout << "Clean time = " << (clean_time/(1000.0)) << " s" << endl;
		cout << "I*(G)/V(G) = " << 100 * ((double)ind_set_size / n) << "%" << endl;

		cout << "Number os SCC = " << n_scc << endl;
		scc_sizes.print();
		
		auto v_impacts = stats_stream<double>("Vertice impact");
		for (auto dd : v_impact) { v_impacts.add(dd); }
		v_impacts.print();
	}
};
// ====================================================

// ============= Data Structures =====================

class imp_treap {

	/*
	Implicit Treap (with min-heap property) data structure such that each element has the <name,value> and implicity key format.
	The elements are sorted by implicity key.

	Supports in O(logN):
	ins(Pos k, <name n, Value v>) : adds element <n,v> at position k
	del(Pos k) : remove element at position k
	max_value() : returns the max value in the tree.
	find_max() : returns name n1 and position pos_max of the element with max_value and n2 the name of the next element (in tree order) after n1, if n2 exists.
	get_pos(name n, cmp_name(), mode) : searchs for node with name n using cmp_name(). It assume that cmp_name() sort names with the same orther that they are in the tree.
	sum_value([p1,p2], Value delta_value) : For each element in position [p1,p2], sums delta to the value, v += delta_value.

	Some important invariants:
	- binary search tree property : left.key < root.key <= right.key
	- min-heap property over priority
	- the node max_value does not include lazy data.
	- names >= -1
	*/

	const static int treap_inf = (1 << 31);
	const static int not_found_code = -2;

public:
	class node {
	public:
		//element variables
		int name, value;
		//subtree variables
		int max_value, lazy_value, count;
		//treap variables
		int priority;
		node *l, *r;

		node(int _name = 0, int _value = 0) :name(_name), value(_value), max_value(_value), lazy_value(0),
			count(1), l(NULL), r(NULL)
		{
			priority = (rand_gen()) % treap_inf;
		}

		// ==== Node Normalization Methods =====
		void update() {
			count = 1;
			max_value = value + lazy_value;

			if (l != NULL) {
				count += l->count;
				max_value = max(max_value, l->max_value + l->lazy_value);
			}

			if (r != NULL) {
				count += r->count;
				max_value = max(max_value, r->max_value + r->lazy_value);
			}
		}

		void lazy_prop() {
			if (lazy_value == 0) return;

			value += lazy_value;
			max_value += lazy_value;

			if (l != NULL) {
				l->lazy_value += lazy_value;
			}
			if (r != NULL) {
				r->lazy_value += lazy_value;
			}

			lazy_value = 0;
		}
		// ===========================

	};

	node *root;

	imp_treap() {
		root = NULL;
	}

private:

	// ==== Rotations ========
	void rr(node *&p) {
		//assume p and p->l != NULL and p not lazy
		node *q = p->l;
		q->lazy_prop();
		p->l = q->r;
		q->r = p;
		p->update(); q->update();
		p = q;
	}

	void lr(node *&p) {
		//assume p and p->r != NULL and p not lazy
		node *q = p->r;
		q->lazy_prop();
		p->r = q->l;
		q->l = p;
		p->update(); q->update();
		p = q;
	}
	// =======================

	// === query parameters ===
	int pos, name, value, pos1, pos2, d_value;
	bool is_lw_bound_srch;
	function<bool(int, int)> *less_func;
	// ========================

	void ins(node *&p, int acc_less = 0) {
		//uses pos, name, value parameters
		//insert <name,value> after position pos
		if (p == NULL) {
			p = new node(name, value);
		}
		else {
			p->lazy_prop();
			int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

			if (pos < my_key) {
				ins(p->l, acc_less);
				if (p->l->priority < p->priority) rr(p);
			}
			else
			{
				ins(p->r, my_key);
				if (p->r->priority < p->priority) lr(p);
			}
		}
		p->update();
	}

	void del_search(node *&p, int acc_less = 0) {
		// uses pos parameter
		// delete element at position pos
		if (p == NULL) return;
		p->lazy_prop();
		int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

		if (my_key == pos)
			del(p);
		else
			if (pos < my_key) del_search(p->l, acc_less);
			else del_search(p->r, my_key);

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

	void sum_value(node *&p, int li, int ri, int acc_less = 0) {
		//for each element in position[pos1, pos2], sums delta to the value, v += delta_value.
		//uses int pos1, int pos2, int d_value
		//node p represents interval [li,ri]
		if (p == NULL) return;
		p->lazy_prop();
		int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

		//[li,ri] in [pos1,pos2]?
		if (li >= pos1 && ri <= pos2) {
			p->lazy_value += d_value;
			return;
		}
		//my_key in [pos1,pos2]?
		if (my_key >= pos1 && my_key <= pos2) {
			p->value += d_value;
		}
		//[li,my_key-1] inter [pos1,pos2] ?
		if (p->l != NULL && pos1 <= (my_key - 1) && pos2 >= li) {
			sum_value(p->l, li, my_key - 1, acc_less);
		}
		//[my_key+1,ri] inter [pos1,pos2]?
		if (p->r != NULL && pos2 >= (my_key + 1) && pos1 <= ri) {
			sum_value(p->r, my_key + 1, ri, my_key);
		}

		p->update();
	}

	pair<int, int> get_pos_cmp(node *&p, int acc_less = 0) {
		// uses int name,vector<int>& pi_inv, char mode
		// if is_lower_bound returns the rightmost node x such thar pi_inv(x) <= pi_inv(name)
		// if is_upper_bound returns the leftmost node such that pi_inv(x) >= pi_inv(name) 
		// returns pair<position, value>
		if (p == NULL) {
			return make_pair(not_found_code, not_found_code);
		}
		p->lazy_prop();
		int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

		if (p->name == name) {
			return make_pair(my_key, p->value);
		}

		//comp btw name and p->name 
		if ((*less_func)(name, p->name)) {
			// if is lower bound search, then p and p->r subtree are not acceptable
			// if not, and name not found, p is better
			auto ret = get_pos_cmp(p->l, acc_less);
			if (!is_lw_bound_srch && ret.first == not_found_code) {
				return make_pair(my_key, p->value);
			}
			return ret;
		}
		else {
			// if upper bound search , then p and p->l subtree are not acceptable
			// if lower bound search and name not found, p is better
			auto ret = get_pos_cmp(p->r, my_key);
			if (is_lw_bound_srch && ret.first == not_found_code) {
				return make_pair(my_key, p->value);
			}
			return ret;
		}

	}

	pair<int, int> get_pos(node *&p, int acc_less = 0) {
		// find node by position. returns <name,value>
		//uses pos
		if (p == NULL) {
			return make_pair(not_found_code, not_found_code);
		}
		p->lazy_prop();
		int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

		if (my_key == pos) {
			return make_pair(p->name, p->value);
		}

		if (pos < my_key) {
			return get_pos(p->l, acc_less);
		}
		else {
			return get_pos(p->r, my_key);
		}
	}

	//returns name n1 and position pos_max of the element with max_value and 
	//n2 the name of the next element(in tree order) after n1, if n2 exists.
	void find_max(node *&p, pair<pair<int, int>, int> &ret, int acc_less = 0) {

		if (p == NULL) return;
		p->lazy_prop();
		int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);

		if (p->value == p->max_value) {
			//max node founde
			ret.first.first = p->name; ret.first.second = my_key; ret.second = not_found_code;
			//looking for next
			if (p->r != NULL) {
				node *next = p->r;
				while (next->l != NULL)
				{
					next = next->l;
				}
				ret.second = next->name;
			}
			return;
		}

		if (p->l != NULL && (p->l->max_value + p->l->lazy_value == p->max_value)) {
			//max is in left subtree
			find_max(p->l, ret, acc_less);
			//test if p is n2
			if (ret.second == not_found_code) ret.second = p->name;
		}
		else {
			//max from right subtree
			find_max(p->r, ret, my_key);
		}
	}

	void clear(node *&p) {
		if (p == NULL) return;
		clear(p->l); clear(p->r);
		delete p; p = NULL; return;
	}

	void print(node *&p, string &ret) {
		if (p == NULL) return;
		p->lazy_prop();

		print(p->l, ret);
		ret = ret + "(" + std::to_string(p->name) + " , " + std::to_string(p->value) + ")" + " ";
		print(p->r, ret);
	}

public:

	void ins(int _pos, int _name, int _value) {
		pos = _pos; name = _name; value = _value;
		ins(root);
	}

	void del(int _pos) { pos = _pos; del_search(root); }

	int max_value() {
		if (root == NULL) return (-treap_inf);
		root->lazy_prop();
		return root->max_value;
	}

	//returns name n1 and position pos_max of the element with max_value and 
	//n2 the name of the next element(in tree order) after n1, if n2 exists.
	void find_max(pair<pair<int, int>, int> &ret) {
		ret.second = not_found_code;
		return find_max(root, ret);
	}

	// if is_lower_bound returns the rightmost node x such thar pi_inv(x) <= pi_inv(name)
	// if is_upper_bound returns the leftmost node such that pi_inv(x) >= pi_inv(name) 
	// returns pair<position, value>
	pair<int, int> get_pos(int _name,
		function<bool(int, int)> &_less_func
		, bool _is_lw_bound_srch) {
		name = _name; less_func = &_less_func; is_lw_bound_srch = _is_lw_bound_srch;
		return get_pos_cmp(root);
	}

	pair<int, int> get_pos(int _pos) {
		pos = _pos;
		return get_pos(root);
	}

	//for each element in position[pos1, pos2], sums delta to the value, v += delta_value.
	void sum_value(int _pos1, int _pos2, int _d_value) {
		pos1 = _pos1; pos2 = _pos2; d_value = _d_value;
		return sum_value(root, 1, root->count);
	}

	~imp_treap() { clear(); }
	void clear() { clear(root); }

	int count() { return root == NULL ? 0 : root->count; }

	string to_string() {
		string ret = "";
		print(root, ret);
		return ret;
	}

};

// ====================================================

// ============== Instance and Solution representation ===============

struct node
{
	node() {};
	node(int _idx, int _w) { idx = _idx; w = _w; }
	node(int _idx, int _w, bool _inv) { idx = _idx; w = _w; inv = _inv; }
	
	int idx,w;
	bool inv = false;
	
	
	static bool comp_idx(node n1, node n2) {
		return n1.idx < n2.idx;
	}

};

class instance;

class tarjan_scc {
	vector<int> idx, lowlink;
	vector<bool> onStack;	
	stack<int> ss;
	int idx_act;
	const instance * s;
	
public:
	vector<vector<int>> scc_list;
	void solve(const instance& _s);
private:
	void tarjan_scc::scc(int v);
};

class instance {
	string inst_filepath;
	std::default_random_engine my_rand_gen = std::default_random_engine(0);

	public:
		int n;
		vector<vector<node>> g;		
		inst_stats i_stats;
		long long fo_offset_cleaned = 0;
		
		// ==================== Initialization ==================
		//O(n^2 + m*log(delta))
		void initialize(string filepath) { 

			// ===== Open instance file =================
			ifstream inst_file;		
			inst_file.open(filepath, ios::in);
			inst_filepath = filepath;
			inst_file >> n ;
			i_stats.init(n);
			// ======  Create g from data stream ========
			g = vector<vector<node>>(n);
			int i, j;
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {					
					int w;
					inst_file >> w;					
					if (i == j) continue;

					if (w != 0) {
						i_stats.edge_number++;

						// edge i->j , weight w
						g[i].push_back(node(j, w));
						g[j].push_back(node(i, w, true));
					}
				}
			}
			
			inst_file.close();
			//===========
			clean();
			
		}

		//O(m*log(delta)) [could be O(m) usging radix sort]
		void clean() {
			auto startTime = get_time::now();
			fo_offset_cleaned = 0;

			/* w(i,j) = max(w(i,j),w(j,i)) - min(w(i,j),w(j,i)) */

			int i,j;
			for (i = 0; i < n; i++) {
				// sort g[i] using idx (could be radix sort)
				sort(g[i].begin(), g[i].end(),node::comp_idx);
				
				// recreate g[i]
				auto ng = vector<node>(); ng.reserve(g[i].size());
				for (j = 0; j < g[i].size(); j++) {
					int u = i, v = g[i][j].idx;

					if (j + 1 < g[i].size() && g[i][j + 1].idx == v) {
						// case v->u and u->v (delete one)
						int w_uv = (!g[i][j].inv) ? g[i][j].w : g[i][j+1].w;
						int w_vu = g[i][j].inv ? g[i][j].w : g[i][j + 1].w;
						int minCost = min(w_uv, w_vu);
						w_uv -= minCost; w_vu -= minCost;					
						
						//saving objective function offset for future record 
						if (u < v) { fo_offset_cleaned += minCost; }

						if (w_uv > 0)
						{
							ng.push_back(node(v, w_uv));
							i_stats.edge_number_clean++;
							i_stats.v_impact[u] += w_uv;
						}

						if (w_vu > 0)
						{
							ng.push_back(node(v, w_vu, true));
							i_stats.v_impact[u] += w_vu;
						}
						
						j++;
					}else{
						// case only v->u xor u->v
						ng.push_back(node(v, g[i][j].w,g[i][j].inv));
						i_stats.v_impact[u] += g[i][j].w;

						if(!g[i][j].inv)
							i_stats.edge_number_clean++;
					}
				}
				g[i] = ng;				 
			}
			auto finishTime = get_time::now();
			i_stats.clean_time += to_time((finishTime - startTime));
		}
		
		// ============ Algorithms ==============================
		void compute_scc_stats() {
			tarjan_scc scc_solver;
			scc_solver.solve((*this));

			i_stats.n_scc = scc_solver.scc_list.size();
			for (auto scc : scc_solver.scc_list) {
				i_stats.scc_sizes.add(scc.size());
			}
		}

		void ind_set_analysis(vector<int> pi) {
			/* how many vertex can be extrated to form an independent set ?
				*/
			vector<bool> is_inative(n);

			//repeat : choose one and exclude neighborhood
			for (auto idx : pi) {
				if (is_inative[idx]) continue;

				is_inative[idx] = true;
				i_stats.ind_set_size++;
				
				for (auto nb : g[idx]) {
					is_inative[nb.idx] = true;
				}
			}
		}
		// ================== Utilities =============================
		void print_g(int lim = 10000) {
			cout << "n = " << n << endl;
			for (int i = 0; i < n; i++) {
				if (i >= lim) break;
				cout << "N(" << i << ") = {";
				for (auto nb : g[i]) {
					cout << "(" << nb.idx << "," << nb.w << "," << (nb.inv ? "in" : "out") << ")";
				}
				cout << "}" << endl;
			}
		}

		long long compute_fo(vector<int> &pi_inv) {
			long long fo = 0;
			for (auto v = 0; v < n; v++) {
				for (auto nb : g[v]) {
					auto u = nb.idx;
					if (!nb.inv && pi_inv[v] < pi_inv[u])
						fo += nb.w;
				}
			}
			return fo + fo_offset_cleaned;
		}

		bool hard_check(vector<int> &pi, long long fo_calc) {
			vector<int> pi_inv(pi.size());
			for (int ii = 0; ii < pi.size(); ii++) pi_inv[pi[ii]] = ii;

			//open instance file and assert that pi has fo_calc value
			// ===== Open instance file =================
			ifstream inst_file;
			inst_file.open(inst_filepath, ios::in);
			inst_file >> n;
			int i, j; long long fo_file = 0;
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					int w;
					inst_file >> w;
					if (i == j) continue;
					if (w != 0) {
						// edge i->j , weight w
						if (pi_inv[i] < pi_inv[j]) fo_file += w;
					}
				}
			}
			inst_file.close();
			//check if fo´s are equal
			if (fo_calc != fo_file) {
				cout << "Error - objective function not equal to file : " << fo_calc << " != "
					<< fo_file << endl;
				exit(1);
			}

		}

		void restart_pi_gen(int seed = 0) {
			my_rand_gen = std::default_random_engine(seed);
		}

		vector<int> generate_random_pi() {
			vector<int> pi(n);
			for (int i = 0; i < n; i++) pi[i] = i;
			shuffle(pi.begin(), pi.end(), my_rand_gen);
			return pi;
		}

};

//TODO improve solution classes design
class solution {
protected:
	instance *my_inst;
public:
	int n;
	vector<vector<node>> g;
	vector<int> pi, pi_inv;
	long long fo_online = 0;
	
	//O(m*log(delta))
	void force_ng_order() {
		auto cmp_func = [&](node n1, node n2) -> bool { return pi_inv[n1.idx] < pi_inv[n2.idx]; };
		fo_online = 0;
		for (int i = 0; i < n; i++){
			sort(g[i].begin(), g[i].end(), cmp_func);
			for (int j = 0; j < g[i].size(); j++)
				if (!g[i][j].inv && pi_inv[i] < pi_inv[g[i][j].idx])
					fo_online += g[i][j].w;
		}
	}

	long long get_fo() {
		return (fo_online + my_inst->fo_offset_cleaned);
	}

	//O(log(|N(v)|)
	//locate vertex u in N(v) assuming that N(v) is sorted by pi_inv and pi_inv(u) = pinv_u
	//if u not in N(v) return idx for the first >= 
	//return index in N(v)
	int locate(int v, int pinv_u) {

		auto resp = lower_bound(g[v].begin(), g[v].end(), pinv_u,
			[&](node& x, int _p_inv_u) -> bool { return pi_inv[x.idx] < _p_inv_u; } //x < u?
		);

		return resp - g[v].begin();
	}

	bool check() {
		vector<bool> is_active(n, false);
		for (auto i = 0; i < n; i++) {
			if (is_active[pi[i]]) return false;
			is_active[pi[i]] = true;
			if (pi_inv[pi[i]] != i) return false;
		}
		return true;
	}

	//prints check() and permutation
	void print(bool print_pi) {
		cout <<
			" Check = " << (check() ? "Ok" : "No")
			<< endl;

		if (print_pi)
		{
			cout << "Pi = {";
			for (auto idx : pi) cout << idx << " , ";
			cout << "}" << endl;
		}
	}
};

class solution_1 : public solution {
	bool builded_matrix = false;

	void build_matrix() {
		if (builded_matrix) return; //never changes

		int i, j;
		matrix.resize(n);
		for (i = 0; i < n; i++) matrix[i].resize(n);

		for (i = 0; i < n; i++) {
			for (j = 0; j < g[i].size(); j++) {
				int v = i, u = g[i][j].idx, w = g[i][j].w;
				if (!g[i][j].inv) matrix[v][u] = w;
				else matrix[u][v] = w;
			}
		}

		builded_matrix = true;
	}

public:
	vector<vector<int>> matrix;
	bool is_matrix_builded() { return builded_matrix; }

	void init(instance& inst, vector<int>& pi, bool _build_matrix = false) {
		my_inst = &inst;
		n = inst.n;
		g = inst.g; //copy
		if (_build_matrix)
			build_matrix();
		reset_pi(pi);
	}

	void reset_pi(vector<int>& pi) {
		this->pi = pi;
		pi_inv = vector<int>(n);
		int i;
		for (i = 0; i < n; i++) {
			pi_inv[pi[i]] = i;
		}
		fo_online = 0;
		force_ng_order();
	}

	//O(m*log(delta))
	//idx_pointer[i][j] = position of vertex i in the neighbhorhood of vertex g[i][j].idx
	//before_v[v] = last j | pi´(g[v][j].idx) < pi´(v)
	void compute_aux_pointers(vector<vector<int>>& idx_pointer, vector<int>& before_v) {		
		idx_pointer = vector<vector<int>>(n);
		before_v = vector<int>(n);

		for (int v = 0; v < n; v++) {
			before_v[v] = -1;
			idx_pointer[v] = vector<int>(g[v].size());

			for (int j = 0; j < g[v].size(); j++) {
				int u = g[v][j].idx;
				//u before v?
				if (pi_inv[u] < pi_inv[v]) before_v[v] = j;

				//idx of v in N(u)?
				auto v_idx = locate(u, pi_inv[v]);
				idx_pointer[v][j] = v_idx;
			}
		}
	}

};

class solution_2 : public solution {
	
public:	
	vector<imp_treap> trees;
	vector<int> cost;
	function<bool(int, int)>  comp_less_func;

	//O(m*log(delta))
	void init(instance& inst, vector<int>& pi) {
		my_inst = &inst;
		n = inst.n;
		g = inst.g; //copy
		reset_pi(pi);
	}
	
	//O(m*log(delta))
	void reset_pi(vector<int>& pi) {
		this->pi = pi; pi_inv.resize(n);		
		trees.resize(n); cost.resize(n);

		for (int i = 0; i < n; i++) {
			pi_inv[pi[i]] = i;
		}
		fo_online = 0;

		//this cmp function takes account dummy vertix with name = -1
		comp_less_func = [&](int v1, int v2) -> bool {
			if (v1 == -1) return true;
			if (v2 == -1) return false;
			return pi_inv[v1] < pi_inv[v2];
		};

		force_ng_order();
		for(int v=0;v<n;v++)
			make_neighborhood_trees(v);
	}
	
	//O(m*log(delta))
	void make_neighborhood_trees(int v) {
		//assume neighborhoods in g[v] are ordered
		trees[v].clear();
		cost[v] = 0;
		
		int i,w = 0,cv = 0;
		node * nb;
		for (i = 0; i < g[v].size(); i++) {
			nb = &g[v][i];
			if (!nb->inv) w += nb->w;
			if (!nb->inv && (pi_inv[v] < pi_inv[nb->idx])) cv += nb->w;
			if (nb->inv && (pi_inv[v] > pi_inv[nb->idx])) cv += nb->w;
		}
		cost[v] = cv;

		//inserting fake first vertex <-1,w>
		trees[v].ins(1, -1, w);
		//inserting other vertices
		for (i = 0; i < g[v].size(); i++) {
			nb = &g[v][i];
			w += nb->w * (nb->inv ? 1 : -1 );
			trees[v].ins(i + 2, nb->idx, w);
		}
	}

};

// =================== Local Search ========================================

class ls_engine {
	
	vector<vector<int>> ins_delta_mtrx; //delta insert matrix (used for double moves)

	// ====== behavior variables =========
	bool use_binary_search = false; //only valid if algorithm is bs_array_alg
	bool use_memory = false; //memory (at now) is for debug propose only		
	bool do_second_order_srch = false;

public:
	
	// ======= meta data (improve design for unordered_maps <-> enum names) ============
	enum move_strg { first, best , best_heap} my_move_strg;
	static unordered_map<move_strg, string> move_strg_name;
	enum round_order_strg { idx_order, rand_round_order } my_round_order_strg;
	static unordered_map<round_order_strg, string> round_order_strg_name;
	enum ls_algorithms_option {
		basic_alg,array_alg, bs_array_alg,
		array_idx_pointers_alg, tree_alg} my_algorithm;
	static unordered_map<ls_algorithms_option, string> ls_algorithm_name;	
	// ====== statistics variables =========
	stats_stream<long> one_round_time_stats = stats_stream<long>("round time [ms]");
	stats_stream<long> one_round_time_stats2 = stats_stream<long>("round(2) time [ms]");
	long total_time = 0, n_ls_moves = 0, n_ls_moves2 = 0;
	vector<pair<pair<int, int>,int>> moves_memory;
private:
	void init_variables() {
		total_time = 0;
		n_ls_moves = 0;
		n_ls_moves2 = 0;
		moves_memory.clear();
		one_round_time_stats = stats_stream<long>("round time [ms]");
		one_round_time_stats2 = stats_stream<long>("round(2) time [ms]");
	}

public:	
	// ============
	ls_engine() {
		//improve design for unordered_maps - gambi level Hight
		move_strg_name[move_strg::best] = "best_move";
		move_strg_name[move_strg::best_heap] = "best_heap_move";
		move_strg_name[move_strg::first] = "first_move";

		round_order_strg_name[round_order_strg::idx_order] = "fixed_order";
		round_order_strg_name[round_order_strg::rand_round_order] = "rand_order";		

		ls_algorithm_name[ls_algorithms_option::basic_alg] = "basic";
		ls_algorithm_name[ls_algorithms_option::array_alg] = "normal_array";
		ls_algorithm_name[ls_algorithms_option::bs_array_alg] = "bs_array";
		ls_algorithm_name[ls_algorithms_option::array_idx_pointers_alg] = "idx_pointers_array";
		ls_algorithm_name[ls_algorithms_option::tree_alg] = "tree";
	}

	void config(ls_algorithms_option ls_algorithm , move_strg mv_strg, 
		round_order_strg _r_ord_strg, bool _use_mem = false, 
		bool do_second_order_srch_ = false) {
		
		if (ls_algorithm == ls_algorithms_option::bs_array_alg)
			use_binary_search = true;
		else
			use_binary_search = false;

		my_algorithm = ls_algorithm;
		my_move_strg = mv_strg;
		my_round_order_strg = _r_ord_strg;
		use_memory = _use_mem;
		do_second_order_srch = do_second_order_srch_;

	}

	string signature(char sep = ' ' , bool omit_roud_order = false) {
		//ls_name, move_str, round_order_strg
		string ret = ls_algorithm_name[my_algorithm];
		if (do_second_order_srch) ret += "(order2)";
		ret += sep + move_strg_name[my_move_strg];
		if(!omit_roud_order) ret += sep + round_order_strg_name[my_round_order_strg];
		
		return ret;
	}

	void run_ls(solution_1 &s) {
		switch (my_algorithm)
		{
			case basic_alg:
				ls_basic_insert(s);
				break;
			case array_alg:				
			case bs_array_alg:
				ls_array_insert(s);
				break;
			case array_idx_pointers_alg:
				ls_insert_idx_store(s);
				break;
			case tree_alg:
			default:
				cout << "ERROR - incorrect solution representation for tree.";
				exit(1);
				break;
		}
	}

	void run_ls(solution_2 &s) {
		switch (my_algorithm)
		{
			case tree_alg:
					ls_insert_tree(s);
					break;
			default:
				cout << "ERROR - no algorithm found." << endl;
				exit(1);
				break;
		}

	}

private:
	// ======= PI manipulation ==============
	void update_pi(solution &s, int v, int u) {
		//update pi and pi_inv for insert(v,u)
		bool fw = s.pi_inv[v] <= s.pi_inv[u];
		int i;

		// ======= Update pi and pi_inv ========================
		if (fw) {
			for (i = s.pi_inv[v]; i < s.pi_inv[u]; i++) {
				s.pi[i] = s.pi[i + 1];
				s.pi_inv[s.pi[i]] = i;
			}
		}
		else {
			for (i = s.pi_inv[v]; i > s.pi_inv[u]; i--) {
				s.pi[i] = s.pi[i - 1];
				s.pi_inv[s.pi[i]] = i;
			}
		}
		s.pi[i] = v; s.pi_inv[v] = i;
		//=======================================================
	}

	void update_pi_dm(solution &s, int v, int u, int v_fw, int u_bw) {
		//update pi and pi_inv for di(v,u,v_fw,u_bw)
		int pv = s.pi_inv[v], pvv = s.pi_inv[v_fw],
			pu = s.pi_inv[u], puu = s.pi_inv[u_bw];
		int tmp;

		for (int i = s.pi_inv[v]+1; i < s.pi_inv[u]; i++) {
			if (i < puu) {
				//bw
				s.pi[i - 1] = s.pi[i];
				s.pi_inv[s.pi[i]] = i - 1;
			}else if(i <= pvv){
				//nothing to do
			}else{
				//fw
				break;
			}
		}
		for (int i = pu - 1; i >= pvv + 1; i--) {
			s.pi[i + 1] = s.pi[i];
			s.pi_inv[s.pi[i]] = i + 1;
		}
		
		s.pi_inv[v] = pvv + 1; s.pi[s.pi_inv[v]] = v;
		s.pi_inv[u] = puu - 1; s.pi[s.pi_inv[u]] = u;

	}

	// ======== Locas Search Algorithms (TODO Improve Design) =============

	void ls_array_insert(solution_1& s) {
		init_variables();

		auto start_ls_time = get_time::now();
		// ========== LS start ==============

		int v, u, n = s.n;
		bool stop_ls = false;
		vector<int> round_order(n); for (int i = 0; i < n; i++) round_order[i] = i;

		do {
			auto start_round_time = get_time::now();
			//======== round start =================

			int best_v = -1, best_v_j = -1, best_improve, i, j;
			int max_shuffle_idx = n - 1;
			
			for (int vv = 0; vv < n; vv++) {
				if (my_round_order_strg == round_order_strg::rand_round_order) {
					v = select_shuffle(round_order, max_shuffle_idx);
				}else
					v = vv;

				// exploring N_insert(v)
				auto start_round_time = get_time::now();

				int delta, best_j, best_delta,
					before_v = -1, dv = s.g[v].size();

				//locate v order in N(v)
				if (!use_binary_search) {
					//O(dv)
					for (j = 0; j < dv; j++) {
						u = s.g[v][j].idx;
						if (s.pi_inv[u] > s.pi_inv[v]) break;
						before_v = j;
					}
				}
				else {
					//O(log(dv))
					before_v = s.locate(v, s.pi_inv[v]) - 1;
				}

				best_j = -1; delta = 0;
				//can improve from insert(v,u) | pi´(u) > pi´(v)? (v in front of u)
				for (j = before_v + 1; j < dv; j++) {
					delta += s.g[v][j].w * (s.g[v][j].inv ? 1 : -1);
					if (best_j == -1 || delta > best_delta) {
						best_j = j; best_delta = delta;
					}
				}

				//can improve from insert(v,u) | pi´(u) < pi´(v)? (v before u)
				delta = 0;
				for (j = before_v; j >= 0; j--) {
					delta += s.g[v][j].w * (s.g[v][j].inv ? -1 : 1);
					if (best_j == -1 || delta > best_delta) {
						best_j = j; best_delta = delta;
					}
				}

				if (best_j == -1) {
					//case: N(v) = empty
				}
				else if (best_v == -1 || best_delta > best_improve) {
					//case: best move from v is the new best
					best_v = v;
					best_improve = best_delta;
					best_v_j = best_j;
				}
				else {
					//case: insert(v,*) improves but its worse than previous
					//nothing to do
				}

				if ((my_move_strg == move_strg::first) && best_v != -1 && best_improve > 0) break;
			}

			if (best_v == -1 || best_improve <= 0) {
				//cannot improve insert(*,*) => local minumum
				stop_ls = true;
			}
			else {				
				// ===== Apply insert(v,u) =======
				v = best_v;
				j = best_v_j;
				u = s.g[v][j].idx;
				bool fw = s.pi_inv[u] > s.pi_inv[v];
				s.fo_online += best_improve;

				if (use_memory)
					moves_memory.push_back(make_pair(make_pair(v, u),best_improve));

				//======== Reconstruct pi and pi_inv in range [pi´(v) , pi´(u)] ========
				auto pinv_v_old = s.pi_inv[v];
				update_pi(s, v, u);
				// ====================================================================

				//foreach u in N(v) => adjust v in N(u)
				for (auto nb_v : s.g[v]) {
					u = nb_v.idx;
					auto du = s.g[u].size();

					//locate v in N(u)
					if (use_binary_search) {
						auto tmp = s.pi_inv[v];
						s.pi_inv[v] = pinv_v_old;
						auto idx_near_v = s.locate(u, pinv_v_old);
						s.pi_inv[v] = tmp;
						i = (idx_near_v < du && s.g[u][idx_near_v].idx == v) ? idx_near_v : 0;
						i = (idx_near_v + 1 < du && s.g[u][idx_near_v + 1].idx == v) ? idx_near_v + 1 : i;
						i = (idx_near_v - 1 < du && idx_near_v - 1 >= 0 && s.g[u][idx_near_v - 1].idx == v) ? idx_near_v - 1 : i;
					}
					else {
						for (i = 0; v != s.g[u][i].idx; i++);
					}

					//swaps
					if (fw)
					{
						while (i + 1 < du && s.pi_inv[s.g[u][i].idx] > s.pi_inv[s.g[u][i + 1].idx]) {
							swap(s.g[u][i + 1], s.g[u][i]);
							i++;
						}
					}
					else {
						while (i - 1 >= 0 && s.pi_inv[s.g[u][i].idx] < s.pi_inv[s.g[u][i - 1].idx]) {
							swap(s.g[u][i - 1], s.g[u][i]);
							i--;
						}
					}
				}
			}

			//=========== end of round ============
			auto end_round_time = get_time::now();
			one_round_time_stats.add(to_time(end_round_time - start_round_time));			
			if(!stop_ls) n_ls_moves++;

		} while (!stop_ls);


		// ===== end of ls ========
		auto end_ls_time = get_time::now();
		total_time = to_time(end_ls_time - start_ls_time);
	}

	void ls_insert_idx_store(solution_1& s) {
		init_variables();

		// ====== Computing aux idx pointers ========
		//idx_pointer[i][j] = position of vertex i in the neighbhorhood of vertex g[i][j].idx
		vector<vector<int>> idx_pointer;

		//before_v[v] = last j | pi´(g[v][j].idx) < pi´(v)		
		vector<int> before_v;

		s.compute_aux_pointers(idx_pointer, before_v);
		// ============================================

		// ========== LS Start ===================================================

		auto start_ls_time = get_time::now();

		int v, u, n = s.n;
		bool stop_ls = false;
		vector<int> round_order(n); for (int i = 0; i < n; i++) round_order[i] = i;

		do {
			auto start_round_time = get_time::now();
			//======== round start =================

			int best_v = -1, best_v_j = -1, best_improve, i, j, u;
			int max_shuffle_idx = n - 1;

			for (int vv = 0; vv < n; vv++) {
				if (my_round_order_strg == round_order_strg::rand_round_order) {
					v = select_shuffle(round_order, max_shuffle_idx);
				}
				else
					v = vv;

				// Exploring moves insert(v,*)
				auto start_round_time = get_time::now();

				int delta, best_j, best_delta,
					dv = s.g[v].size();

				best_j = -1; delta = 0;
				//can improve from insert(v,u) | pi´(u) > pi´(v)? (v in front of u)
				for (j = before_v[v] + 1; j < dv; j++) {
					delta += s.g[v][j].w * (s.g[v][j].inv ? 1 : -1);
					if (best_j == -1 || delta > best_delta) {
						best_j = j; best_delta = delta;
					}
				}

				//can improve from insert(v,u) | pi´(u) < pi´(v)? (v before u)
				delta = 0;
				for (j = before_v[v]; j >= 0; j--) {
					delta += s.g[v][j].w * (s.g[v][j].inv ? -1 : 1);
					if (best_j == -1 || delta > best_delta) {
						best_j = j; best_delta = delta;
					}
				}

				if (best_j == -1) {
					//case: N(v) = empty
				}
				else if (best_v == -1 || best_delta > best_improve) {
					//case: best move from v is the new best
					best_v = v;
					best_improve = best_delta;
					best_v_j = best_j;
				}
				else {
					//case: insert(v,*) improves but its worse than previous
					//nothing to do
				}

				if ((my_move_strg == move_strg::first) && best_v != -1 && best_improve > 0) break;
			}

			if (best_v == -1 || best_improve <= 0) {
				//cannot improve insert(*,*) => local minumum
				stop_ls = true;
			}
			else {
				// Apply insert(v,u)
				v = best_v;
				j = best_v_j;
				u = s.g[v][j].idx;
				bool fw = s.pi_inv[u] > s.pi_inv[v];
				s.fo_online += best_improve;

				if (use_memory)
					moves_memory.push_back(make_pair(make_pair(v, u),best_improve));

				//updating before_v[v]
				before_v[v] = fw ? j : j - 1;

				//reconstruct pi and pi_inv in range [pi´(v) , pi´(u)]
				update_pi(s, v, u);

				//foreach u in N(v) => adjust v in N(u)
				auto dv = s.g[v].size();
				for (int i_nb_v = 0; i_nb_v < dv; i_nb_v++) {
					u = s.g[v][i_nb_v].idx;
					auto du = s.g[u].size();

					//locate v idx in N(u)
					i = idx_pointer[v][i_nb_v];
					int vi, idx_u_vi;

					//keep N(u) sorted and adjust idx_pointer for moving vertices
					if (fw)
					{
						//swaps i -> i+1
						while (i + 1 < du && s.pi_inv[s.g[u][i].idx] > s.pi_inv[s.g[u][i + 1].idx]) {
							vi = s.g[u][i].idx;
							idx_u_vi = idx_pointer[u][i];
							idx_pointer[vi][idx_u_vi]++; //moving fw

							vi = s.g[u][i + 1].idx;
							idx_u_vi = idx_pointer[u][i + 1];
							idx_pointer[vi][idx_u_vi]--; //moving bw

							swap(idx_pointer[u][i], idx_pointer[u][i + 1]);
							swap(s.g[u][i + 1], s.g[u][i]);

							i++;
						}
						//updating before_v[u]
						if (before_v[u] >= 0 && s.pi_inv[s.g[u][before_v[u]].idx] >= s.pi_inv[u])
							before_v[u]--;
					}
					else {
						//swaps i -> i-1
						while (i - 1 >= 0 && s.pi_inv[s.g[u][i].idx] < s.pi_inv[s.g[u][i - 1].idx]) {
							vi = s.g[u][i].idx;
							idx_u_vi = idx_pointer[u][i];
							idx_pointer[vi][idx_u_vi]--; //moving bw

							vi = s.g[u][i - 1].idx;
							idx_u_vi = idx_pointer[u][i - 1];
							idx_pointer[vi][idx_u_vi]++; //moving fw

							swap(idx_pointer[u][i], idx_pointer[u][i - 1]);

							swap(s.g[u][i - 1], s.g[u][i]);
							i--;
						}
						//updating before_v[u]
						if (before_v[u] + 1 < du &&
							s.pi_inv[s.g[u][before_v[u] + 1].idx] <= s.pi_inv[u])
							before_v[u]++;
					}
				}
			}

			//=========== end of round ============
			auto end_round_time = get_time::now();
			one_round_time_stats.add(to_time(end_round_time - start_round_time));
			if (!stop_ls) n_ls_moves++;

		} while (!stop_ls);


		// ===== end of ls ========
		auto end_ls_time = get_time::now();
		total_time = to_time(end_ls_time - start_ls_time);
	}

	void ls_basic_insert(solution_1& s) {
		if (!s.is_matrix_builded()) {
			cout << "Error - not builded matrix" << endl;
			exit(1);
		}

		init_variables();

		if (do_second_order_srch) {
			ins_delta_mtrx.resize(s.n);
			for(int i=0;i<s.n;i++)
				if (ins_delta_mtrx[i].size() != s.n) {
					ins_delta_mtrx[i].resize(s.n);
				}
				else break;
		}

		auto start_ls_time = get_time::now();
		// ========== LS start ==============

		int v, u, n = s.n;
		bool stop_ls = false;
		vector<int> round_order(n); for (int i = 0; i < n; i++) round_order[i] = i;

		do {
			auto start_round_time = get_time::now();
			auto end_round_time = get_time::now();
			//======== round start =================

			int best_v = -1, best_v_u, best_improve, i, j;
			int max_shuffle_idx = n - 1,ii;

			for (ii = 0; ii < n; ii++) {				
				if (my_round_order_strg == round_order_strg::rand_round_order) {
					v = select_shuffle(round_order, max_shuffle_idx);
				}
				else
					v = ii;
				
				i = s.pi_inv[v];
				auto delta_cost = 0, best_v_improve = -I32_INF, best_u = -1;
				//best move from v to the right
				for (j = i + 1; j < n; j++) {
					u = s.pi[j];
					delta_cost += -s.matrix[v][u] + s.matrix[u][v];
					if (delta_cost > best_v_improve) {
						best_v_improve = delta_cost;
						best_u = u;
					}
					if (do_second_order_srch) {						
						ins_delta_mtrx[v][u] = delta_cost;
					}
				}
				//move from v to left
				delta_cost = 0;
				for (j = i - 1; j >= 0; j--) {
					u = s.pi[j];
					delta_cost += s.matrix[v][u] - s.matrix[u][v];
					if (delta_cost > best_v_improve) {
						best_v_improve = delta_cost;
						best_u = u;
					}
					if (do_second_order_srch) {
						ins_delta_mtrx[v][u] = delta_cost;
					}
				}
				//save best v so far
				if (best_v == -1 || best_v_improve > best_improve) {
					best_v = v; best_improve = best_v_improve;
					best_v_u = best_u;
				}

				if ((my_move_strg == move_strg::first) && best_v != -1 && best_improve > 0) break;
			}

			if (best_v == -1 || best_improve <= 0) {
				//cannot improve insert(*,*) => local minumum of order 1

				//=========== stop counting round time
				end_round_time = get_time::now();				
				//====================================

				bool second_ord_improved = false;
				if (do_second_order_srch) {
					//test second order neighborhood
					second_ord_improved = do_double_move(s, round_order);					
				}
				if (second_ord_improved) {
					n_ls_moves2++;
				}else
					stop_ls = true;

			}
			else {
				// ===== Apply insert(v,u) =======
				v = best_v;
				u = best_v_u;
				s.fo_online += best_improve;

				if (use_memory)
					moves_memory.push_back(make_pair(make_pair(v, u), best_improve));

				update_pi(s, v, u);

				//=========== stop counting round time
				end_round_time = get_time::now();
				//====================================

			}
			one_round_time_stats.add(to_time(end_round_time - start_round_time));

			//test time limit
			long long dummy;
			if (tl_clock_check(dummy))
				stop_ls = true;
			
			if (!stop_ls) n_ls_moves++;

		} while (!stop_ls);

		// ===== end of ls ========
		auto end_ls_time = get_time::now();
		total_time = to_time(end_ls_time - start_ls_time);
	}

	bool do_double_move(solution_1& s, vector<int>& round_order) {

		int v, u, n = s.n;
		bool found_improve = false;

		auto start_round_time = get_time::now();
		//======== round start =================

		int best_v = -1,
			best_v_u = -1, best_improve = -1, i, j,
			best_v_fw_found = -1, best_u_bw_found = -1;
		int max_shuffle_idx = n - 1, vv;

		//searching double move
		for (vv = 0; vv < n; vv++) {
			if (my_round_order_strg == round_order_strg::rand_round_order) {
				v = select_shuffle(round_order, max_shuffle_idx);
			}
			else
				v = vv;

			//Test if exists fw double move from v				
			i = s.pi_inv[v];
			//test fw double move with some vertex u 
			for (j = i + 1; j < n; j++) {
				u = s.pi[j];

				if (s.matrix[u][v] <= 0) continue;

				//test di(v,u,*,*)
				int max_online_bw_insert_cost = -1,
					u_bw_online = -1;

				//looking for v_fw
				for (int k = i + 1; k < j; k++) {
					int v_fw = s.pi[k];

					//save best bw edge from u
					if (u_bw_online == -1 || ins_delta_mtrx[u][v_fw] > max_online_bw_insert_cost) {
						u_bw_online = v_fw;
						max_online_bw_insert_cost = ins_delta_mtrx[u][u_bw_online];
					}

					//di(v,u,v_fw,u_bw_online) improve solution?
					if (s.matrix[v_fw][v] <= 0) continue;
					int improve_di = ins_delta_mtrx[v][v_fw] + max_online_bw_insert_cost + s.matrix[u][v];
					if (improve_di > 0) {
						//improve move found
						best_improve = improve_di;
						best_v_fw_found = v_fw;
						best_u_bw_found = u_bw_online;
						best_v = v; 
						best_v_u = u;
						break;
					}
				}
				if (best_improve > 0)break;
			}
			if (best_improve > 0) break;
		}


		if (best_v == -1 || best_improve <= 0) {
			//cannot improve di(*,*,*,*) => second order local minumum
			//nothing todo =(			
			found_improve = false;
		}
		else {
			// ===== Apply double  =======
			int v_fw = best_v_fw_found, u_bw = best_u_bw_found;
			v = best_v; u = best_v_u;
			
			s.fo_online += best_improve;
			update_pi_dm(s, v, u, v_fw, u_bw);
			found_improve = true;
		}

		//=========== end of round ============
		auto end_round_time = get_time::now();
		one_round_time_stats2.add(to_time(end_round_time - start_round_time));
		
		return found_improve;
	}

	//update N(g[v][iw]) for insert(v,u)
	void update_tree_neighborhood(solution_2 &s, int v, int u,bool fw, int iw) {
		const node * vw = &s.g[v][iw];		
		//locate v in N(vw)
		auto v_block = s.trees[vw->idx].get_pos(v, s.comp_less_func, true);
		auto v_pos = v_block.first;
		//locate u region
		auto u_block = s.trees[vw->idx].get_pos(u, s.comp_less_func, fw);
		auto u_pos = u_block.first; auto u_w = u_block.second;

		//if v and u are in the same region (relative to N(vw)) nothing to do
		if (v_pos == u_pos) return;
		
		//======= computing v new value and delta_value change in intermediate vertices 
		int before_u_weight, delta_value = 0, v_value;
		if (!fw) {
			before_u_weight = s.trees[vw->idx].get_pos(u_pos - 1).second;
			//d_value = -(vw -> vi) + (vi -> vw)
			delta_value = vw->inv ? -(vw->w) : vw->w;
			v_value = before_u_weight + delta_value;
		}else{//fw			
			//d_value = -(vi -> vw) + (vw -> vi)
			delta_value = vw->inv ? vw->w : -(vw->w);
			v_value = u_w;
		}
		//====================

		//remove v block from its position
		s.trees[vw->idx].del(v_pos);
		//insert v block in new position
		s.trees[vw->idx].ins(u_pos - 1, v, v_value);
		//change values in vertices between insert operation
		if (fw)
			s.trees[vw->idx].sum_value(v_pos, u_pos - 1, delta_value);
		else
			s.trees[vw->idx].sum_value(u_pos+1, v_pos, delta_value);
	}
	
	void ls_insert_tree(solution_2 &s) {
		init_variables();
		auto start_ls_time = get_time::now();
		// ========== LS start ==============
		bool stop_ls = false;
		int v, u, n = s.n, i;
		vector<int> round_order(n); for (i = 0; i < n; i++) round_order[i] = i;

		//initializing heap if its the case
		set<pair<int,int>,greater<pair<int,int>>> max_heap;	
		if (my_move_strg == best_heap) {
			for (v = 0; v < n; v++) {
				max_heap.insert(make_pair(s.trees[v].max_value() - s.cost[v], v));				
			}
		}

		do
		{			
			auto start_round_time = get_time::now();
			//======== Round start ===========

			//======= Searching for improving move from vertex v =======
			int best_v = -1, best_improve, max_shuffle_idx = n - 1;
			
			if (my_move_strg == best_heap) {				
				auto max_element = max_heap.begin();
				best_v = max_element->second;
				best_improve = max_element->first;
			}
			else {
				for (i = 0; i < n; i++) {
					if (my_round_order_strg == round_order_strg::rand_round_order) {
						v = select_shuffle(round_order, max_shuffle_idx);
					}
					else
						v = i;

					auto best_improve_v = s.trees[v].max_value() - s.cost[v];
					if (best_v == -1 || best_improve_v > best_improve) {
						best_v = v; best_improve = best_improve_v;
					}
					if (my_move_strg == move_strg::first && best_v != -1 && best_improve > 0) break;
				}
			}
			//========================================

			if (best_v == -1 || best_improve <= 0) { //!moves with 0 improvment are not supported!
				//=== Local Minimum Found
				stop_ls = true;
			}
			else {

				if (my_move_strg == best_heap) {
					//update heap
					max_heap.erase(make_pair(best_improve, best_v));
					max_heap.insert(make_pair(0, best_v));
				}
				
				//=== Selecting improving move
				v = best_v; u = -1; bool fw = false;

				pair<pair<int, int>, int> best_move_info;
				s.trees[v].find_max(best_move_info);
				int v1 = best_move_info.first.first,
					p1 = best_move_info.first.second,
					v2 = best_move_info.second;
				
				//best move puts v in region after vertex v1
				if (v1 == -1) { //v1 == -1 represents region before every vertex in v neigbhorhood
					u = v2; fw = false;

					//assert
					if (s.pi_inv[v] < s.pi_inv[u]) {
						cout << "Erro ls-tree - best move = dummy" << endl;
						exit(1);
					}
				}
				else {
					if (s.pi_inv[v] < s.pi_inv[v1] + 1) {
						fw = true; u = v1;
					}else{
						fw = false; u = v2;
					}
				}

				//=========== Apply Insert(v,u) improving move ========
				s.fo_online += best_improve;
				if (use_memory) {
					moves_memory.push_back(make_pair(make_pair(v,u),best_improve));
				}

				for (i = 0; i < s.g[v].size();i++) {
					int vw = s.g[v][i].idx, sign;

					if (my_move_strg == best_heap) {
						//update heap element
						auto old_cost = s.trees[vw].max_value() - s.cost[vw];
						max_heap.erase(make_pair(old_cost, vw));
					}

					// ===== Update N(g[v][i]) =====
					update_tree_neighborhood(s, v, u, fw, i); //log(du)
					
					// ===== Update cost[vw] =========											
					if (fw && s.pi_inv[v] < s.pi_inv[vw] && s.pi_inv[vw] <= s.pi_inv[u]) {
						sign = (s.g[v][i].inv ? 1 : -1);
						s.cost[v] += s.g[v][i].w * sign;
						s.cost[vw] += s.g[v][i].w * sign;
					}else if(!fw && s.pi_inv[u] <= s.pi_inv[vw] && s.pi_inv[vw] < s.pi_inv[v]){
						sign = (s.g[v][i].inv ? -1 : 1);
						s.cost[v] += s.g[v][i].w * sign;
						s.cost[vw] += s.g[v][i].w * sign;
					}
					
					if (my_move_strg == best_heap) {
						//update heap element
						auto new_cost = s.trees[vw].max_value() - s.cost[vw];
						max_heap.insert(make_pair(new_cost, vw));
					}
					
				}

				//update pi and pi_inv
				update_pi(s, v, u);				
			}
			
			//======== Round end ===========
			auto end_round_time = get_time::now();
			one_round_time_stats.add(to_time(end_round_time - start_round_time));
			if (!stop_ls) n_ls_moves++;

		} while (!stop_ls);

		// ========== LS end ================
		auto end_ls_time = get_time::now();
		total_time = to_time(end_ls_time - start_ls_time);
	}

};
//TODO improve design
unordered_map<ls_engine::move_strg, string> ls_engine::move_strg_name;
unordered_map<ls_engine::round_order_strg, string> ls_engine::round_order_strg_name;
unordered_map<ls_engine::ls_algorithms_option, string> ls_engine::ls_algorithm_name;
//TODO improve design
typedef tuple<ls_engine::ls_algorithms_option, ls_engine::move_strg,
ls_engine::round_order_strg, bool, bool > ls_conf_type;

// ==================== S.C.C Algorithm =======================================
void tarjan_scc::solve(const instance & _s) {
		s = &_s;
		idx = vector<int>(s->n);
		lowlink = vector<int>(s->n);
		onStack = vector<bool>(s->n);
		ss = stack<int>();
		idx_act = 1;
		scc_list.clear();

		for (int v = 0; v < s->n; v++) {
			if (idx[v] == 0)
				scc(v);
		}

	}

void tarjan_scc::scc(int v) {
		idx[v] = idx_act;
		lowlink[v] = idx_act;
		idx_act++;
		ss.push(v);
		onStack[v] = true;

		for (auto nb : s->g[v]) {
			if (nb.inv) continue;
			
			int u = nb.idx;
			if (idx[u] == 0) { //down edge
				scc(u);
				lowlink[v] = min(lowlink[v], lowlink[u]);
			}
			else if (onStack[u]) { //up edge
				lowlink[v] = min(lowlink[v], idx[u]);
			}
		}

		if (lowlink[v] == idx[v]) {
			//v is scc root
			int i_comp = scc_list.size();
			scc_list.push_back(vector<int>());
			
			int u;
			do {
				u = ss.top(); ss.pop();
				onStack[u] = false;
				scc_list[i_comp].push_back(u);				
			} while (u != v);
			
		}

	}
// ==========================================================================

//====================== Instance Analisys ==================================

void do_inst_analisys(instance& inst) {
	//========== Inst Analysis ===========
	//OLD
	//Print instance stats
	//inst_0.i_stats.print();

	inst.compute_scc_stats();

	vector<int> pi_0(inst.n);
	for (int i = 0; i < inst.n; i++)pi_0[i] = i;
	sort(pi_0.begin(), pi_0.end(),
	[&](int u, int v) -> bool {return inst.g[u].size() < inst.g[v].size(); });

	inst.ind_set_analysis(pi_0);
	//========================================

}

// ==========================================================================

// ================= Instance Selection =============================

class instance_selector {
	bool is_unique_inst = false, generated = false;
	string unique_inst_name;
	
public:
	string insts_folder_path = "C:\\Users\\Lucas\\Cursos\\PucRio\\Metaheuristicas\\Projeto1\\";
	 
	int n_a, d_a, i_a, i, j, k;
	vector<int> * nv, * dv, * iv;
	string inst_name, path;

	instance_selector(
		vector<int>& n_sizes,
		vector<int>& d_sizes,
		vector<int>& i_sizes
	) {
		nv = &n_sizes;
		dv = &d_sizes;
		iv = &i_sizes;
		i = j = k = 0;
		k--;

	}

	instance_selector(string _unique_inst_name) {
		is_unique_inst = true;
		this->unique_inst_name = _unique_inst_name;
	}

private:
	string generate_path() {
		int n = (*nv)[i], d = (*dv)[j], ii = (*iv)[k];
		string ns = to_string(n);
		while (ns.length() < 4) ns = "0" + ns;
		string ds = to_string(d);
		while (ds.length() < 3) ds = "0" + ds;
		string is = to_string(ii);

		inst_name = "n" + ns + "d" + ds + "-" + is;
		path = insts_folder_path + "n" + ns + "\\" + inst_name;
		return path;
	}
public:
	string getNext() {
		if (is_unique_inst) {
			if (generated) return "";
			generated = true;

			return insts_folder_path + unique_inst_name;
		}


		k++;
		if (k >= iv->size()) {
			k = 0;
			j++;
			if (j >= dv->size()) {
				j = 0;
				i++;
				if (i >= nv->size()) {
					return "";
				}
			}
		}
		n_a = (*nv)[i]; d_a = (*dv)[j]; i_a = (*iv)[k];
		return generate_path();
	}

	string signature(char sep = ' ') {
		string inst_signature =
			to_string(n_a) + sep + to_string(d_a) + sep + to_string(i_a);
		return inst_signature;
	}
};

// ========================= Experiments =======================


// === Experiment 1 - Collect time data from basic ls algorithms (changing strategies)
//OLD
void do_experiment_1() {
	// =======================================
	string exp_name = "Experiment 1 - Collect time data from basic ls algorithms (changing strategies)";
	cout << "==== " << exp_name << " =====" << endl;
	auto exp_begin_time = get_time::now();		
	reset_rand(0);

	// ======= Instance selection config ===========
	vector<int> n_sizes({
		500,
		1000
		//,2000
		//,3000
		//,4000
	});

	vector<int> dens_sizes({
		1
		,5
		,10
		,50
		//,100
	});

	vector<int> inst_i_sizes({
		1
		, 2
		//, 3
		//, 4
		//, 5
	});

	int n_ls_trials = 3;

	instance_selector inst_selec(n_sizes, dens_sizes, inst_i_sizes);
	// ======================================

	ls_engine::move_strg all_mv_strg[] =
	{ 
		ls_engine::move_strg::best , 
		ls_engine::move_strg::first 
	};

	ls_engine::round_order_strg all_round_strg[] =
	{ 
		ls_engine::round_order_strg::idx_order, 
		ls_engine::round_order_strg::rand_round_order 
	};

	ls_engine::ls_algorithms_option all_alg_options[] =
	{
		ls_engine::array_alg,
		ls_engine::bs_array_alg,
		ls_engine::array_idx_pointers_alg
		//,ls_engine::tree
	};

	// output line format
	//n , d, i, ls_name, ls_time, mv_s, r_st, m_rt, fo	
	cout << "n , dens , inst_ind , ls_name , move_strg , round_strg , ls_time[s] , round_time_mean[ms] , fo_mean, fo_max" << endl;

	while (true)
	{
		string inst_file = inst_selec.getNext();
		if (inst_file == "") break;

		instance inst;
		inst.initialize(inst_file);

		//Genrating random start permutations
		vector<vector<int>> initial_solutions(n_ls_trials);
		for (int i = 0; i < n_ls_trials; i++) {
			initial_solutions[i] = vector<int>(inst.n);
			for (int j = 0; j < inst.n; j++) initial_solutions[i][j] = j;
			shuffle(initial_solutions[i].begin(), initial_solutions[i].end(), rand_gen);
		}

		// Creating Solution and LocalSearch engine
		solution_1 s;
		ls_engine ls;

		//foreach ls strategy
		for (auto mv_strg : all_mv_strg) {
			for (auto round_strg : all_round_strg) {
				//PRINT
				//n , d, i, ls_name, ls_time, mv_s, r_st, m_rt, fo_mean, best_sol

				string header = to_string(inst_selec.n_a) + " , " + to_string(inst_selec.d_a) + " , " + to_string(inst_selec.i_a);
				string mv_strg_name = ls_engine::move_strg_name[mv_strg];
				string round_strg_name = ls_engine::round_order_strg_name[round_strg];
				auto seed_ls = rand_gen();

				//foreach ls algorithm
				for (auto alg_option : all_alg_options) {

					//local search statistics
					stats_stream<double> ls_time_stream("ls time[s]");
					stats_stream<long>fo_stream("fo");
					stats_stream<double> round_time_stream("round time[ms]");
					stats_stream<int> n_moves_stream("n ls moves");
					string ls_name = ls_engine::ls_algorithm_name[alg_option];

					//foreach trial initial solution
					for (auto init_perm : initial_solutions) {

						//initializing solution with given permutation
						s.init(inst, init_perm);
						//configuring local search 
						ls.config(alg_option, mv_strg, round_strg);
						//rand_gen reset
						reset_rand(seed_ls);
						//local search run
						ls.run_ls(s);

						//checking found solution
						if (!s.check()) {
							cout << "ERRO invalid soluion!" << endl;
							exit(1);
						}

						//collecting data
						round_time_stream.add(ls.one_round_time_stats.get_mean());
						n_moves_stream.add(ls.n_ls_moves);
						fo_stream.add(inst.compute_fo(s.pi_inv));
						ls_time_stream.add(ls.total_time / 1000.0);
					}

					//print output line
					//n , d, i, ls, mv_s, r_st, ls_time, m_rt, fo_mean, best_fo
					cout << header << " , " << ls_name << " , " <<
						mv_strg_name << " , " <<
						round_strg_name << " , " <<
						to_string(ls_time_stream.get_mean()) << " , " <<
						round_time_stream.get_mean() << " , " <<
						to_string(fo_stream.get_mean()) << " , " <<
						to_string(fo_stream.get_min_max().second) << " , "
						<< endl;

				}

			}
		}

	}

	// ======================= End =========================

	cout << "Total time[s] = " << chrono::duration_cast<chrono::seconds>(get_time::now() - exp_begin_time).count() << endl;
}

// === Experiment Verify Correctness of LS Implementation (maybe old)
void do_experiment_verify_ls_algorithm() {

	cout << "Experiment to verify correctness of local searchs movements " << endl;

	vector<int> n_sizes({
		//500
		//,
		1000
		//,2000
		//,3000
		//,4000
	});

	vector<int> dens_sizes({
		//1
		//,5
		//,10
		//,
		//50
		//,
		100
	});

	vector<int> inst_i_sizes({
		1
		//, 2
		//, 3
		//, 4
		//, 5
	});

	instance_selector inst_selec(n_sizes, dens_sizes, inst_i_sizes);

	while (true) {
		string inst_path = inst_selec.getNext();
		if (inst_path == "") break;
		instance inst;
		inst.initialize(inst_path);
		ls_engine ls1;
		ls_engine ls_base;

		solution_1 s1; //change for sol_2 if tree
		s1.init(inst, inst.generate_random_pi());
		solution_1 s2;
		s2.init(inst, s1.pi,true);

		auto alg_option1 = ls_engine::ls_algorithms_option::bs_array_alg; //tree_alg;
		auto move_strg1 = ls_engine::move_strg::best;
		auto round_order_strg1 = ls_engine::round_order_strg::idx_order;

		ls1.config(alg_option1, move_strg1, round_order_strg1, false);
		ls_base.config(ls_engine::basic_alg, move_strg1, round_order_strg1, false);
		auto start_time_ls = get_time::now();
		ls1.run_ls(s1);
		cout << "time to run ls1: " <<
			(to_time(get_time::now() - start_time_ls))/1000.0 << " s" << endl;

		ls_base.run_ls(s2);

		//checking found solution
		if (!s1.check() || !s2.check()) {
			cout << "ERROR invalid soluion!" << endl;
			exit(1);
		}

		auto fo1 = inst.compute_fo(s1.pi_inv);
		auto fo2 = inst.compute_fo(s2.pi_inv);

		bool warning = false;

		if (fo1 != fo2) {
			warning = true;
			cout << "Warning " << fo1 << " != " << fo2 << endl;
		}
		else
			cout << fo1 << "=" << fo2 << endl;

		auto n_print_warn = 0;
		if (ls1.moves_memory.size() == 0) cout << "No memory test" << endl;
		for (int i = 0; i < ls1.moves_memory.size(); i++) {
			if (ls1.moves_memory[i] != ls_base.moves_memory[i]) {
				warning = true; n_print_warn++;
				cout << "Warning (" << i << ") "
					<< pii_to_str(ls1.moves_memory[i].first) << " (" << ls1.moves_memory[i].second << ")"
					<< " != " <<
					pii_to_str(ls_base.moves_memory[i].first) << " (" << ls_base.moves_memory[i].second << ")"
					<< endl;

				if (n_print_warn == 1 && ls1.moves_memory[i].second != ls_base.moves_memory[i].second) {
					cout << "Error - first diff have not the same improve" << endl;
					exit(1);
				}

				if (n_print_warn > 4) {
					cout << "..." << endl;
					break;
				}
			}
		}

		if (warning) {
			//check if solution found by ls_1 is a local minimum in ls_2
			s2.reset_pi(s1.pi);
			ls_base.run_ls(s2);
			if (ls_base.n_ls_moves > 0) {
				cout << "Error - not local minumum!" << endl;
				exit(1);
			}
		}

	}

}

// ==== Random Multi Restart Experiment ======
enum stop_criterion_type {time_limit,n_runs_limit};
void do_experiment_multi_restart(
	vector<ls_conf_type> ls_configs,
	vector<int>& n_sizes, vector<int>& dens_sizes, vector<int>& inst_i_sizes,
	int time_limit_sec = 30, int max_n_runs = 10, 
	stop_criterion_type my_stop_criterion = stop_criterion_type::time_limit
) {
	/*
		The ideia of this experiment is to compare the quality/time of ls´s configurations
		under time limit or number of ls runs.
		
		To do this we run each one of theses ls´s continuously from random start solutions (sprint).
		Until some stop criterion (time limite or # of runs)
		
		For each sprint we collect several data such as:
		- number of runs
		- fo value : max , mean, sd
		- mean number of ls moves : max, mean, sd
		- max_fo value over time
		...
	*/

	string exp_name = "Experiment Random Multi Restart";
	cout << "!" << exp_name << " ";
	if (my_stop_criterion == stop_criterion_type::time_limit) {
		cout << "Time Limit " << to_string(time_limit_sec) << "[s]";
	}
	if (my_stop_criterion == stop_criterion_type::n_runs_limit) {
		cout << "#Runs Limit " << to_string(max_n_runs) << "[s]";
	}
	cout << endl;

	auto exp_begin_time = get_time::now();
	reset_rand(0);

	instance_selector inst_selec(n_sizes, dens_sizes, inst_i_sizes);
	
	//=============================

	/* Output format
	(instance signature) (ls signature)..
	n d i ls_name move_strg rmr_duration number_of_runs fo_max fo_mean fo_sd n_moves_mean n_moves_sd n_moves2_mean mean2_round_time

	(in specific file)
	header
	fo1 t1 ls_time n_moves n_moves2
	fo2 t2 ls_time n_moves	
	*/
	
	//print header
	string header_line =
		"n d i ls_name move_strg rmr_duration number_of_runs fo_max fo_mean fo_sd n_moves_mean n_moves_sd n_moves2_mean mean2_round_time";

	string spec_file_header = "fo time ls_time n_ls_moves n_ls_moves2";
	char sep = ' ';
	cout << header_line << endl;

	//foreach instance selected
	while (true)
	{
		string inst_file = inst_selec.getNext();
		if (inst_file == "") break;
		
		//instance initialization
		instance inst;
		inst.initialize(inst_file);

		solution_1 s1;
		solution_2 s2;
		solution *s;
		ls_engine ls;

		//foreach ls_config
		for (auto ls_config : ls_configs) {
			//set ls config
			ls.config(
				get<0>(ls_config),
				get<1>(ls_config),
				get<2>(ls_config),
				get<3>(ls_config),
				get<4>(ls_config)
			);

			auto alg_opt = get<0>(ls_config);
			bool is_tree_alg = (alg_opt == ls_engine::ls_algorithms_option::tree_alg);
			
			//======= prepare case (ls + inst) string id ==============
				
			string inst_signature = inst_selec.signature(sep);
			string ls_signature = ls.signature(sep,true);
			string case_signature = inst_signature + sep + ls_signature;
			
			//============= Initalizing Solution Representation =================
			inst.restart_pi_gen(); //same sequence of random intial sols for every ls_conf
			//initializing first solution
			if (is_tree_alg) {
				s = &s2;
				s2.init(inst, inst.generate_random_pi());
			}else{
				s = &s1;
				s1.init(inst, inst.generate_random_pi(), 
					alg_opt == ls_engine::ls_algorithms_option::basic_alg);
			}
			//iniatilizing data to collect
			stats_stream<long long> fo_stream("fo",true);
			stats_stream<long long> n_ls_moves_stream("number_of_ls_moves",true);
			stats_stream<long long> round_time_means("means round time");
			stats_stream<long long> round_time_means2("means round time2"); //second neighborhhod order
			stats_stream<long long> n_ls_moves2_stream("n_ls_moves2");

			int n_ls_runs = 0;
			vector<int> best_pi; //stores solution found by this case
			
			//open especific output files (TODO make this optional to run fast)
			std::ofstream case_out(case_signature+".txt");
			case_out << spec_file_header << endl;
			
			// ====== starts random multi restart ========
			bool stop_ls = false; bool check_error_mode = false;
			restart_global_clock(time_limit_sec, my_stop_criterion == stop_criterion_type::time_limit);
			long long duration_rmr_analysis_ms = 0; //ms
			do {
				//apply local search
				is_tree_alg ? ls.run_ls(s2) : ls.run_ls(s1);
				n_ls_runs++;
				//check erros
				if (check_error_mode) {
					if (!s->check()) {//O(n)
						cout << "Error invalid solution found." << endl; exit(1);
					}
				}				
				//collect stats data
				bool is_tl_reached = tl_clock_check(duration_rmr_analysis_ms);				 
				auto fo = s->get_fo(); //O(1)
				//saving best fo found so far
				if (fo_stream.empty() || fo > fo_stream.get_max()) { best_pi = s->pi; } //O(n) copy				
				//output specific data: time_point fo ls_time n_moves n_moves_order2 
				case_out << to_string(duration_rmr_analysis_ms) 
					<< sep << to_string(fo) << sep << to_string(ls.total_time)
					<< sep << to_string(ls.n_ls_moves)
					<< sep << to_string(ls.n_ls_moves2)
					<< endl;
				//save aggregate stats data
				fo_stream.add(fo);
				n_ls_moves_stream.add(ls.n_ls_moves);
				n_ls_moves2_stream.add(ls.n_ls_moves2);
				round_time_means.add(ls.one_round_time_stats.get_mean());
				round_time_means2.add(ls.one_round_time_stats2.get_mean());
				
				//test stop criterion				
				if (my_stop_criterion == stop_criterion_type::time_limit &&
					is_tl_reached) {
					stop_ls = true;
				}
				else if (my_stop_criterion == stop_criterion_type::n_runs_limit &&
					(n_ls_runs >= max_n_runs)) {
					stop_ls = true;
				}else					{
					//generate new start solution
					is_tree_alg ? s2.reset_pi(inst.generate_random_pi()) : s1.reset_pi(inst.generate_random_pi());
				}
			} while (!stop_ls);

			/*
				At this point serveral ls´s have already been run.
				We need to print aggregate data for theses local minimums
			*/

			//======= print data line ===========
			string srmr_duration, snumber_of_runs, sfo_max, sfo_mean, sfo_sd, sn_moves_mean, sn_moves_sd, smean_roud_time;
			srmr_duration = to_string((duration_rmr_analysis_ms / 1000.0));//[s]
			snumber_of_runs = to_string(n_ls_runs);
			sfo_max = to_string(fo_stream.get_max());
			sfo_mean = to_string(fo_stream.get_mean());
			sfo_sd = to_string(fo_stream.get_sd());
			sn_moves_mean = to_string(n_ls_moves_stream.get_mean());
			sn_moves_sd = to_string(n_ls_moves_stream.get_sd());			
			smean_roud_time = to_string(round_time_means.get_mean());//[ms]
			
			//============ Print Case Line =====================
			//n d i ls_name move_strg rmr_duration number_of_runs fo_max fo_mean fo_sd n_moves_mean n_moves_sd n_moves2_mean mean2_round_time
			cout << case_signature << sep << srmr_duration << sep << snumber_of_runs << sep
				<< sfo_max << sep << sfo_mean << sep << sfo_sd << sep << sn_moves_mean 
				<< sep << sn_moves_sd
				<< sep << smean_roud_time
				<< sep << to_string(n_ls_moves2_stream.get_mean())
				<< sep << to_string(round_time_means2.get_mean()) //[ms]		
				<< endl;
			//===================================
			
			//close specific output file
			case_out.close();	

			// ==== end of rmr analysis for a ls_config ====
			// do hard_check on best solution found (TODO make it optional)
			inst.hard_check(best_pi, fo_stream.get_max());			
		}
	}

	// ======================= End of Experiment =========================
	cout << "Total Experiment time[s] = " << chrono::duration_cast<chrono::seconds>(get_time::now() - exp_begin_time).count() << endl;
}



// ==============================================================


// =========== Hard Checking Correct Permutation =============

/* Global Hard Check TODO
void do_hard_check() {
	string output_spec_filepath = "";
	std::ifstream file(output_spec_filepath);
	//get pi and fo in file
	vector<int> pi; long long fo=0;
	
	//open inst file
	int n = 1000, d = 5, ii=1;
	vector<int> ns; ns.push_back(n);
	vector<int> ds; ds.push_back(d);
	vector<int> iis; iis.push_back(ii);
	instance_selector inst_select(ns,ds,iis);
	instance inst;
	inst.initialize(inst_select.getNext());

	//perfm check
	inst.hard_check(pi, fo);
}
*/

// ==============================

int main()
{	
	bool _flag_test_mode = false;

	// =========== IO config =================
	std::ofstream out("out.txt");
	auto coutbuf = std::cout.rdbuf(out.rdbuf());	
	

	// ======= Instance selection config ===========
	vector<int> n_sizes({
		500,
		1000
		,2000
		,3000,
		4000
	});

	vector<int> dens_sizes({
		1,
		5,
		10,//*/
		50
		,100
	});

	vector<int> inst_i_sizes({
		1
		,2
		, 3
		, 4
		, 5
	});

	// ======= LS´s configs selections =======

	vector<ls_conf_type> ls_configs;
	/*
	// bs best rand	
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::bs_array_alg,
			ls_engine::move_strg::best,
			ls_engine::round_order_strg::rand_round_order,
			false)
	);*/
	
	//bs first rand
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::bs_array_alg,
			ls_engine::move_strg::first,
			ls_engine::round_order_strg::rand_round_order,
			false,
			false)
	);//*/
	
	/*tree best rand
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::tree_alg,
			ls_engine::move_strg::best,
			ls_engine::round_order_strg::rand_round_order,
			false)
	);*/
	
	//tree first rand	
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::tree_alg,
			ls_engine::move_strg::first,
			ls_engine::round_order_strg::rand_round_order,
			false,
			false)
	);//*/
	
	/* tree best heap (depecreated)
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::tree_alg,
			ls_engine::move_strg::best_heap,
			ls_engine::round_order_strg::rand_round_order,
			false)
	);*/

	//basic first rand
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::basic_alg,
			ls_engine::move_strg::first,
			ls_engine::round_order_strg::rand_round_order,
			false,
			false
			)
	);//*/

	/*basic(2ord) first rand
	ls_configs.push_back(
		make_tuple(
			ls_engine::ls_algorithms_option::basic_alg,
			ls_engine::move_strg::first,
			ls_engine::round_order_strg::rand_round_order,
			false, // use memory?
			true //2nd order?
		)
	);*/

	do_experiment_multi_restart(ls_configs, n_sizes, dens_sizes, inst_i_sizes,
		30, //time limit
		1, //max runs
		stop_criterion_type::n_runs_limit //stop criterion type
	);
	
	//do_experiment_verify_ls_algorithm();
	
	std::cout.rdbuf(coutbuf);
	
}

