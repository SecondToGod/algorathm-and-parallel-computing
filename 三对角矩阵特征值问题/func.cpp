#include "func.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <time.h>
using namespace std;

const double EPS = 1e-3;//精度

void print(vector<vector<double> > &X){
    cout.precision(6);
    cout.setf(ios::fixed);
    for(int i=0 ;i < X.size();i++){
        for(int j=0;j < X[i].size();j++){
            cout<<X[i][j]<<' ';
        }
        cout<<endl;
    }
    cout<<endl;
}

//转置
void transpose(vector<vector<double> > &A,vector<vector<double> > &T){
    if(A.empty() || A[0].empty()) return;
    int m = A.size();
    int n = A[0].size();
    T.clear();
    T.resize(n,vector<double> (m,0));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            T[j][i] = A[i][j];
        }
    }
}
//随机单位向量
void randUnitVector(int n, vector<double> &v){
    srand(time(NULL));
    v.clear();v.resize(n);
    while(true){
        double r=0;
        for(int i=0;i<n;i++){
            v[i]=rand()*1.0/RAND_MAX-0.5;
            r+=v[i]*v[i];
        }
        r=sqrt(r);
        if(r>EPS){
            for(int i=0;i<n;i++)
                v[i]/=r;
            break;
        }
    }
}
//上三角转置
void transpose(vector<vector<double> > &A){
    int m = A.size();
    for(int i=0;i<m;i++){
        for(int j=i+1;j<m;j++){
            swap(A[i][j],A[j][i]);
        }
    }
}

//(level-3)
void multiply(vector<vector<double> > &A,vector<vector<double> > &B,vector<vector<double> > &C){
    C.clear();
    if(A.empty() || A[0].empty() || B.empty() || B[0].empty()) return ;
    C.resize(A.size(),vector<double> (B[0].size(),0));
    for(int i=0;i<A.size();i++){
        for(int j=0;j<B[0].size();j++){
            C[i][j]=0;
            for(int k=0;k<A[0].size();k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

//(level-2)
void multiply(const vector<vector<double> > &X,const vector<double> &v,vector<double> &res){
    res.clear();
    if(X.empty() || v.empty()) return;
    int m = X[0].size();
    res.resize(m,0);
    for(int i=0;i<m;i++){
        for(int j=0;X[i].size();j++){
            res[i] += X[i][j]*v[j];
        }
    }
}

//(level-1)
double dotProduct(const vector<double> & a, const vector<double> & b){
    double res=0;
    for(int i=0;i<a.size();i++)
        res+=a[i]*b[i];
    return res;
}

//res= A*A'*v
void rightMultiply(SparseMatrix &A, const vector<double> & v, vector<double> & res){
    int m=A.rows;
    int n=A.cols;
    res.clear();
    res.resize(m,0);
    vector<double> w(n,0);
    A.moveFirst();
    while(A.hasNext()){
        Cell c=A.next();
        w[c.col]+=c.value*v[c.row];
    }
    A.moveFirst();
    while(A.hasNext()){
        Cell c=A.next();
        res[c.row]+=c.value*w[c.col];
    }
}

//res= A'*A*v
void leftMultiply(SparseMatrix &A, const vector<double> & v, vector<double> & res){
    int m=A.rows;
    int n=A.cols;
    res.clear();
    res.resize(n,0);
    vector<double> w(m,0);
    A.moveFirst();
    while(A.hasNext()){
        Cell c=A.next();
        w[c.row]+=c.value*v[c.col];
    }
    A.moveFirst();
    while(A.hasNext()){
        Cell c=A.next();
        res[c.col]+=c.value*w[c.row];
    }
}

//C= B'*A
void rightMultiply(const vector<vector<double> > & B,SparseMatrix &A, vector<vector<double> > & C){
    int m=B[0].size();
    int k=B.size();
    int n=A.cols;
    for(int i=0;i<C.size();i++) fill(C[i].begin(),C[i].end(),0);
    A.moveFirst();
    while(A.hasNext()){
        Cell c=A.next();
        for(int i=0;i<m;i++){
            C[c.col][i]+=c.value*B[c.row][i];
        }
    }
}

//C <- A B
void leftMultiply(const vector<vector<double> > & B,SparseMatrix &A, vector<vector<double> > & C){
    int r=B[0].size();
    int n=B.size();
    int m=A.rows;
    C.clear();
    C.resize(m,vector<double>(r,0));
    A.moveFirst();
    while(A.hasNext()){
        Cell c = A.next();
        for(int i=0;i<r;i++){
            C[c.row][i]+=c.value*B[c.col][i];
        }
    }
}

//2范数
double norm(const vector<double> &v){
    double r = 0;
    for(int i=0;i<v.size();i++)
        r += v[i]*v[i];
    return sqrt(r);
}

//归一化
double normalize(vector<double> &v){
    double r = 0;
    for(int i=0;i<v.size();i++)
        r += v[i]*v[i];
    r = sqrt(r);
    if(r>EPS){
        for(int i=0;i<v.size();i++)
            v[i] /= r;
    }
    return r;
}

//数乘
void multiply(vector<double> &v, double d){
    for(int i=0;i<v.size();i++)
        v[i] *= d;
}



//特征方程求解
vector<double> secularEquationSolver(vector<double> &z, vector<double> &D, double sigma){
 
    int n=z.size();
    vector<double> res(n);
    //sort : d_0 < d_1 < ... < d_{n-1}
    vector<int> index;
    vector<double> d(n);
    merge_sort(D,index);
    if(sigma<0)
        reverse(index.begin(),index.end());
    vector<double> b(n);
    for(int i=0;i<n;i++){
        b[i]=z[index[i]];
        d[i]=D[index[i]];
    }

    vector<double> lambda(n);
    for(int i=0;i<n;i++){
        vector<double> delta(d.size());
        for(int j=0;j<delta.size();j++)
            delta[j]=(d[j]-d[i])/sigma;
        double gamma=0;
        if(i+1<n){
            //gamma>1/delta[i+1]
            double A=b[i]*b[i];
            double B=-A/delta[i+1]-1;
            for(int j=0;j<delta.size();j++)
                if(j!=i)
                    B-=b[j]*b[j]/delta[j];
            double C=1;
            for(int j=0;j<delta.size();j++)
                if(j!=i)
                    C+=b[j]*b[j]/delta[j];
            C/=delta[i+1];
            C-=b[i+1]*b[i+1]/delta[i+1];
            gamma=(-B+sqrt(B*B-4*A*C))/(2*A);
        }
        //牛顿法迭代求解
        double diff=1;
        while(diff*diff>EPS){
            double g=0;
            for(int j=0;j<n;j++){
                g-=b[j]*b[j]/((delta[j]*gamma-1)*(delta[j]*gamma-1));
            }
            double f=1;
            for(int j=0;j<n;j++){
                f+=b[j]*b[j]/(delta[j]-1/gamma);
            }
            //f+g(newGamma-gamma)=0
            double newGamma=-f/g+gamma;
            diff=fabs(newGamma-gamma);
            gamma=newGamma;
        }
        lambda[i]=1/gamma*sigma+d[i];
    }

    for(int i=0;i<n;i++)
        res[index[i]]=lambda[i];
    return res;
}

//分治实现
void DCSub(vector<double> &alpha, vector<double> &beta, vector<vector<double> > &Q, vector<double> &D, int start, int end){
    if(start==end){
        Q[start][start]=1;
        D[start]=alpha[start];
        return;
    }else{
        int mid=(start+end)/2;

        //        |    mid        mid+1
        //---------------------------------------------
        //    mid    |    a(mid)      b(mid+1)
        //    mid+1    |    b(mid+1)    a(mid+1)

        alpha[mid]-=beta[mid+1];
        alpha[mid+1]-=beta[mid+1];
        DCSub(alpha,beta,Q,D,start,mid);
        DCSub(alpha,beta,Q,D,mid+1,end);
        //alpha[mid]+=beta[mid+1];
        //alpha[mid+1]+=beta[mid+1];


        int n=end-start+1;
        vector<double> z(n,0);
        for(int i=start;i<=mid;i++)
            z[i-start]=Q[mid][i];
        for(int i=mid+1;i<=end;i++)
            z[i-start]=Q[mid+1][i];

        //calculate eigen systems of matrix D+beta[mid+1]*z*z'

        vector<double> d(n,0);
        for(int i=0;i<n;i++)
            d[i]=D[i+start];

        //secular equation is:
        // 1 + \sum_j \frac{z^2_j}{d_j-\lambda} =0
        vector<double> lambda=secularEquationSolver(z, d, beta[mid+1]);

        
        //eigen vector: 
        //P = (D-\lambda I)^{-1} z
        vector<vector<double> > P(n,vector<double>(n));
        for(int i=0;i<n;i++){//for each eigen value
            vector<double> p(n);
            for(int j=0;j<n;j++)
                p[j]=1.0/(D[j+start]-lambda[i])*z[j];
            normalize(p);
            for(int j=0;j<n;j++)
                P[j][i]=p[j];
        }
        
        vector<vector<double> > oldQ(n,vector<double>(n));
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++){
                oldQ[i][j]=Q[i+start][j+start];
            }
        
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                Q[i+start][j+start]=0;
                for(int k=0;k<n;k++){
                    Q[i+start][j+start]+=oldQ[i][k]*P[k][j];
                }
            }
        }
        //eigen value
        for(int i=0;i<n;i++)
            D[i+start]=lambda[i];
    }
}

//分治三对角
void DCTridiagonal(vector<double> alpha, vector<double> &beta, vector<vector<double> > &Q, vector<double> &D){
    //input: A m*m symmetry tridiagonal matrix
    //output: diagonal matrix D, and orthogonal D, such that A=QRQ'
    int m=alpha.size();
    Q.clear();
    Q.resize(m,vector<double>(m,0));
    D.clear();
    D.resize(m,0);
    DCSub(alpha, beta, Q, D, 0, m-1);
}

//svd分解
void svd(SparseMatrix &A, int r, vector<vector<double> > &U, vector<double> &s, vector<vector<double> > &V, string algo){
    //A=U*diag(s)*V'
    //A:m*n matrix sparse matrix
    //U:m*r matrix, U[i]=i th left singular vector
    //s:r vector
    //V:n*r matrix, V[i]=i th right singular vector
    int m=A.rows;
    int n=A.cols;
   
    //lanczos: A*A'=P*T*P'
    if(m<=n){
        int l=m;
        vector<vector<double> > P(m,vector<double>(l,0));
        vector<double> alpha(l,0);
        vector<double> beta(l,0);
        lanczos(A,P,alpha,beta,l);
        vector<vector<double> > W;
       if(algo=="DC"){
		vector<double> D(l,0);
		vector<vector<double> > Q;
		DCTridiagonal(alpha,beta,Q,D);//调用分治法分解
		//合并排序
		vector<int> index;
		merge_sort(D,index);
		reverse(index.begin(),index.end());
		W.resize(l,vector<double>(l));
		for(int i=0;i<l;i++)
			for(int j=0;j<l;j++)
				W[i][j]=Q[i][index[j]];
        }
        U.clear();
		U.resize(m,vector<double>(l));
        multiply(P,W,U);
        for(int i=0;i<U.size();i++)
            U[i].resize(r);
        V.clear();V.resize(n,vector<double>(r));
        rightMultiply(U,A,V);
        s.clear();s.resize(r,0);
        for(int i=0;i<r;i++){
            for(int j=0;j<n;j++)
                s[i]+=V[j][i]*V[j][i];
            s[i]=sqrt(s[i]);
            if(s[i]>EPS){
                for(int j=0;j<n;j++)
                V[j][i]/=s[i];
            }
        }
    }else{
        int l=n;
        vector<vector<double> > P(n,vector<double>(l,0));
        vector<double> alpha(l,0);
        vector<double> beta(l,0);
        lanczosT(A,P,alpha,beta,l);
        
        vector<vector<double> > W;
		if(algo=="DC"){
		vector<double> D(l,0);
		vector<vector<double> > Q;
		DCTridiagonal(alpha,beta,Q,D);
		//need sort
		vector<int> index;
		merge_sort(D,index);
		reverse(index.begin(),index.end());
		W.resize(l,vector<double>(l));
		for(int i=0;i<l;i++)
			for(int j=0;j<l;j++)
				W[i][j]=Q[i][index[j]];
        }
        V.clear();V.resize(n,vector<double>(l,0));
        U.clear();U.resize(m,vector<double>(r,0));
        multiply(P,W,V);
        for(int i=0;i<V.size();i++)
            V[i].resize(r);
        leftMultiply(V,A,U);
        s.clear();s.resize(r,0);
        for(int i=0;i<r;i++){
            for(int j=0;j<m;j++)
                s[i]+=U[j][i]*U[j][i];
            s[i]=sqrt(s[i]);
            if(s[i]>EPS){
                for(int j=0;j<m;j++)
                U[j][i]/=s[i];
            }
        }
    }
}
void lanczosT(SparseMatrix &A, vector<vector<double> > &P, vector<double> &alpha, vector<double> &beta, unsigned int rank){
    //P'*A'*A*P = T = diag(alpha) + diag(beta,1) + diag(beta, -1)
    //P=[p1,p2, ... , pk]
    rank=min(A.cols,min(A.rows,rank));
    vector<double> p;
    unsigned int m=A.rows;
    unsigned int n=A.cols;
    vector<double> prevP(n,0);
    randUnitVector(n,p);
    P.clear();
    P.resize(n,vector<double>(rank,0));
    vector<double> v;
    alpha.clear();alpha.resize(rank);
    beta.clear();beta.resize(rank);
    beta[0]=0;
    for(int i=0;i<rank;i++){
        for(int j=0;j<p.size();j++){
            P[j][i]=p[j];
        }
        leftMultiply(A, p, v);
        alpha[i]=dotProduct(p,v);
        if(i+1<rank){
            for(int j=0;j<n;j++)
                v[j]=v[j]-beta[i]*prevP[j]-alpha[i]*p[j];
            beta[i+1]=norm(v);
            prevP=p;
            for(int j=0;j<n;j++)
                p[j]=v[j]/beta[i+1];
        }
    }
}

void lanczos(SparseMatrix &A, vector<vector<double> > &P, vector<double> &alpha, vector<double> &beta, unsigned int rank){
    //P'*A*A'*P = T = diag(alpha) + diag(beta,1) + diag(beta, -1)
    //P=[p1,p2, ... , pk]
    rank=min(A.cols,min(A.rows,rank));
    vector<double> p;
    unsigned int m=A.rows;
    unsigned int n=A.cols;
    vector<double> prevP(m,0);
    randUnitVector(m,p);
    P.clear();
    P.resize(m,vector<double>(rank,0));
    vector<double> v;
    alpha.clear();alpha.resize(rank);
    beta.clear();beta.resize(rank);
    beta[0]=0;
    for(int i=0;i<rank;i++){
        for(int j=0;j<p.size();j++){
            P[j][i]=p[j];
        }
        rightMultiply(A, p, v);
        alpha[i]=dotProduct(p,v);
        if(i+1<rank){
            for(int j=0;j<m;j++)
                v[j]=v[j]-beta[i]*prevP[j]-alpha[i]*p[j];
            beta[i+1]=norm(v);
            prevP=p;
            for(int j=0;j<m;j++)
                p[j]=v[j]/beta[i+1];
        }
    }
}
