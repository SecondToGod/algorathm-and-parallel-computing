#include <iostream>
#include <cstdlib>
#include "func.h"

int main(){
    cout.precision(6);
    cout.setf(ios::fixed);
	//srand(time(NULL));
    SparseMatrix A;
    A.rows = 10;
    A.cols = 8;
    int rank = 4;
    //生成随机矩阵
    for(int i=0;i<A.rows;i++){
        for(int j=0;j<A.cols;j++){
            A.cells.push_back(Cell(i,j,i+j));
        }
    }

    vector<vector<double> >U,V;
    vector<double> s;
	//string method = 'DC';//定义分解方法QR或者分治
    svd(A,rank,U,s,V,"DC");
    cout<<"[U,S,V]=svd(A,r)"<<endl;
    cout<<"r = "<<rank<<endl;
    cout<<"A = "<<endl;
    A.moveFirst();
    for(int i=0;i<A.rows;i++){
        for(int j=0;j<A.cols;j++){
            cout<<A.next().value<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    cout<<"U = "<<endl;
    print(U);

    cout<<"s = "<<endl;
    for(int i=0;i<s.size();i++){
        for(int j=0;j<s.size();j++){
            if(i == j) cout<<s[i]<<' ';
            else cout<<0.0<<' ';
        }
        cout<<endl;
    }
    cout<<endl;
    cout<<"V = "<<endl;
    print(V);
	system("pause");
    return 0;
}