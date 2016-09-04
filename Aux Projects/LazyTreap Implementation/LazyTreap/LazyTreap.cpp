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
#include <functional>

using namespace std;

// ============= Utility =====================================
#define D_INF (DBL_MAX)
#define I32_INF (1<<31)
auto rand_gen = std::default_random_engine(0);
void reset_rand(int seed) {
	rand_gen = std::default_random_engine(seed);
}

string pii_string(pair<int, int> pii) {		
	return ("(" + to_string((_Longlong)pii.first) + " , " + to_string((_Longlong)pii.second) + ")");
}
//============================================================


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

	const static int treap_inf = (1<<31);
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
		node *l,*r;

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
	int pos, name, value, pos1,pos2, d_value;
	bool is_lw_bound_srch;
	function<bool(int, int)> *less_func;
	// ========================

	void ins(node *&p, int acc_less = 0) {
		//uses pos, name, value parameters
		//insert <name,value> after position pos
		if (p == NULL) {
			p = new node(name,value);
		}
		else {
			p->lazy_prop();
			int my_key = acc_less + 1 + (p->l != NULL ? p->l->count : 0);
			
			if (pos < my_key) {
				ins(p->l,acc_less);
				if (p->l->priority < p->priority) rr(p);
			}
			else
			{
				ins(p->r,my_key);
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

	pair<int,int> get_pos_cmp(node *&p, int acc_less = 0){
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
		}else{
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
	void find_max(node *&p, pair<pair<int,int>,int> &ret, int acc_less = 0) {
				
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
	void find_max(pair<pair<int,int>,int> &ret) { 
		ret.second = not_found_code;
		return find_max(root, ret); 
	}

	// if is_lower_bound returns the rightmost node x such thar pi_inv(x) <= pi_inv(name)
	// if is_upper_bound returns the leftmost node such that pi_inv(x) >= pi_inv(name) 
	// returns pair<position, value>
	pair<int,int> get_pos(int _name,
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
		return sum_value(root,1,root->count);
	}
	
	~imp_treap() { clear(); }
	void clear() { clear(root); }

	int count() { return root == NULL ? 0 : root->count; }

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

	void ins(int _pos, int _name, int _value) {
		
		v.insert(v.begin() + _pos, make_pair(_name, _value));
	}

	void del(int _pos) {
		_pos--;
		v.erase(v.begin() + _pos);
	}

	int max_value() {
		int ret = (-I32_INF);
		for (auto pp : v) { ret = max(ret,pp.second); }
		return ret;
	}

	void find_max(pair<pair<int, int>, int> &ret) {
		ret.second = -2;
		auto maxv = max_value();
		for (int i = 0; i < v.size(); i++) {
			if (v[i].second == maxv) {
				ret.first.first = v[i].first; ret.first.second = i+1;
				ret.second = (i + 1 < v.size() ? v[i+1].first : -2);
				break;
			}
		}
	}

	void sum_value(int pos1, int pos2, int d_value) {
		for (int i = 0; i < v.size(); i++) {
			if (i+1 >= pos1 && i+1 <= pos2) {
				v[i].second += d_value;
			}
		}
	}

	int count() {
		return v.size();
	}

	string to_string() {
		string ret = "";
		for (int i = 0; i < v.size(); i++) {
			ret = ret + pii_string(v[i]) + " ";
		}
		return ret;
	}
};

int main() {
	
	string op;
	imp_treap tree;
	array_imp array;

	int n = 10,i, w_max = 100;
	for (i = 1; i <= n; i++) {
		int w = 1 + rand_gen() % w_max;
		tree.ins(0, i, w);
		array.ins(0, i, w);
	}

	cout << tree.to_string() << endl;
	cout << array.to_string() << endl;
	cout << "======== Test Begin =======" << endl;

	int K, N, V, D, P1, P2;
	bool allways_print_array = true;
	pair<int, int> ret;
	string tret, aret;
	
	string ops[] = {"I","R","MV","FM","SV","P"};

	bool rand_mode = true;

	while (true) {
		if (!rand_mode)
			cin >> op;
		else
			op = ops[rand_gen()%6];

		if (array.count() == 0) op = "I";

		cout << "operacao = " << op << endl;

		tret = ""; aret = "";

		if (op == "X") break;

		//- insert I P N V
		if (op == "I") {
			if(!rand_mode)
				cin >> P1 >> N >> V;
			else {
				P1 = rand_gen() % (array.count() + 1);
				N = rand_gen() % 10000;
				V = rand_gen() % 10000;
			}

			tree.ins(P1,N,V);
			array.ins(P1,N,V);
		}
		
		//- remove R P
		if (op == "R") {
			if(!rand_mode)
				cin >> P1;
			else {
				P1 = rand_gen() % (array.count());
				P1++;
			}
			tree.del(P1);
			array.del(P1);
		}

		//max_value MV
		if (op == "MV") {
			tret = to_string(tree.max_value());
			aret = to_string(array.max_value());
		}

		//find_max FM
		if (op == "FM") {
			pair<pair<int, int>, int> ret;

			tree.find_max(ret);
			tret = to_string(ret.first.first) + to_string(ret.first.second) + to_string(ret.second);

			array.find_max(ret);
			aret = to_string(ret.first.first) + to_string(ret.first.second) + to_string(ret.second);
		}

		//- sum_value SV P1 P2 DV
		if (op == "SV") {
			if (!rand_mode) {
				cin >> P1 >> P2 >> D;
			}else{
				P1 = (rand_gen() % (array.count())) + 1;
				P2 = (rand_gen() % (array.count())) + 1;
				if (P1 > P2) swap(P1, P2);
				D = rand_gen() % 10000;
			}
			
			tree.sum_value(P1, P2, D);
			array.sum_value(P1, P2, D);
		}

		//7 - print P
		if (op == "P") {
			string tstr = tree.to_string(); string astr = array.to_string();
			cout << "Tree = " << tstr << endl;
			cout << "Array = " << astr << endl;

			if (tstr != astr) {
				cout << "Erro" << endl;
			}
		}

		if (tret != "" || aret != "") {
			cout << "retornos" << endl;
			cout << tret << endl << aret << endl;
			cout << "===========" << endl;

			if (tret != aret) {
				cout << "Erro" << endl; exit(1);
			}
		}
		
		if (op != "P" && allways_print_array) {
			cout << "===== array ======" << endl;
			cout << array.to_string() << endl;
			cout << "==================" << endl;
		}
		cout << endl;
	}

	return 0;
}