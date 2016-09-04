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
#include <sstream>
#include <string>

using namespace std;

int main() {

	string filepath = "C:\\Users\\Lucas\\Cursos\\PucRio\\Metaheuristicas\\Projeto1\\Projeto1_LOP\\Projeto1_LOP\\1R 2ord\\out.txt";
	//C:\Users\Lucas\Cursos\PucRio\Metaheuristicas\Projeto1\Projeto1_LOP\Projeto1_LOP\1R 2ord
	ifstream file;
	file.open(filepath, ios::in);
	string line;

	ofstream outfile;
	outfile.open("out.txt_");
	
	while (getline(file, line)) {
		istringstream ss(line);
		string palavra = "";
		bool order_match = false;
		bool first = true;

		while (ss >> palavra) {			
			if (first) {
				first = false;
				outfile << palavra;
			}else{
				if (order_match) {
					order_match = false;
					outfile << palavra;
				}else
					outfile << " " << palavra;
			}

			if (palavra == "basic(order")
				order_match = true;
			
		}
		outfile << endl;
	}

	//char c;
	//cin >> c;
	outfile.close();
	file.close();
	return 0;
}
