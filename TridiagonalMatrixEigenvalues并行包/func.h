#ifndef FUNC_H
#define FUNC_H
#include <vector>
#include <string>
using namespace std;

class Cell{//矩阵元素
public:
    unsigned int row;
    unsigned int col;
    double value;
    Cell():row(0),col(0),value(0){};
    Cell(int r,int c,double v):row(r),col(c),value(v){};
};

class SparseMatrix{//解析矩阵
public:
    unsigned int rows;
    unsigned int cols;
    vector<Cell> cells;

    int cellID;
    //序列化读取数据
    void moveFirst(){
        cellID=0;
    }
    bool hasNext(){
        return cellID < cells.size();
    }
    Cell next(){
        return cells[cellID++];
    }
};


void resolve(SparseMatrix &A,int r,vector<vector<double> > &U,vector<double> &s,vector<vector<double> > &V);

void print(vector<vector<double> > &A);//打印矩阵

void lanczos(SparseMatrix &A, vector<vector<double> > &P, vector<double> &alpha, vector<double> &beta, unsigned int rank);


template <class T>
void combine(vector<T> &v,int left,int m,int right,vector<int> &index){//归并排序
    vector<T> tempv(v.begin()+left,v.begin()+right+1);
    vector<int> tempindex(index.begin()+left,index.begin()+right+1);

    int left_size = m-left+1;
    int size = right-left+1;
    int middle = m-left+1;
    int i = 0,j = middle,k = left;
    while(i<left_size && j<size){
        if(tempv[i]<=tempv[j]){
            v[k] = tempv[i];
            index[k] = tempindex[i];
            k++;
            i++;
        }else{
            v[k] = tempv[j];
            index[k] = tempindex[j];
            k++;
            j++;
        }
    }
    while(i<left_size){
        v[k] = tempv[i];
        index[k] = tempindex[i];
        k++;
        i++;
    }
}

template<class T>
void merge_sort(vector<T> &v,int left,int right,vector<int> &index){
    if(left<right){
        int m = (left+right)/2;
        merge_sort(v,left,m,index);
        merge_sort(v,m+1,right,index);
        combine(v,left,m,right,index);
    }
}

template<class T>
void merge_sort(vector<T> v,vector<int> &index){
    index.clear();
    index.resize(v.size());
    for(int i=0;i<v.size();i++) index[i]=i;
    merge_sort(v,0,v.size()-1,index);
}
#endif