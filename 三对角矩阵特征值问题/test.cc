#include <iostream>
#include <time.h>
#include <cstdlib>
#include <unistd.h>

using namespace std;
int main(){
	clock_t start, finish;
	start = clock();
	sleep(3);
    //cout<<time(NULL);
    //cout<<srand(time(NULL));
	finish = clock();
	cout << (double)(finish - start) / CLOCKS_PER_SEC;
    system("pause");
    return 0;
}