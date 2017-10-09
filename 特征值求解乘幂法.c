/*作者:朱京乔 计算机网络信息中心
日期：9月12日
方法：乘幂法
输入文件由以下格式构成，第一行是矩阵的维数
以下各行是矩阵的值，比如
3 3
1.0 1.0 0.5
1.0 1.0 0.25
0.5 0.25 2.0
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void multiple(double**A,double *V,int dim_x,int dim_y);
double max(double *V,int dim);
void div_matrix(double *V,int dim,double m);

int main(){
    FILE *file = fopen("dengjin.txt","r");
    int dim_x,dim_y;
    double **A,*V;
    double miu0 = 0,miu1 = 10;/*just make sure to enter the loop*/
    fscanf(file,"%d %d",&dim_x,&dim_y);

/*load in data*/
    A = (double **)malloc(sizeof(double *)*dim_x);
    V = (double *)malloc(sizeof(double)*dim_y);
    for(int i=0;i<dim_x;i++)
        A[i] = (double *)malloc(sizeof(double)*dim_y);

    for(int i=0;i<dim_x;i++)
        for(int j=0;j<dim_y;j++)
            fscanf(file,"%lf",&A[i][j]);

    for(int i=0;i<dim_y;i++)
        V[i] = 1;/*initialing a vector with any value*/

    while(fabs(miu1-miu0) >= 1E-8){
        multiple(A,V,dim_x,dim_y);
        miu0 = miu1;
        miu1 = max(V,dim_y);
        div_matrix(V,dim_y,miu1);
        for(int i=0;i<dim_y;i++)
            printf("%10.8lf ",V[i]);
        puts("");
    }
    printf("Eigenvalue: %10.8lf\n",miu1);
    //delocating


    free(V);
    for(int i=0;i<dim_y;i++)
        free(A[i]);
    free(A);

    return EXIT_SUCCESS;
}

void multiple(double**A,double *V,int dim_x,int dim_y){
    double *tmp = (double *)malloc(sizeof(double)*dim_y);
    for(int i=0;i<dim_y;i++)
        tmp[i] = 0;

    for(int i=0;i<dim_x;i++)
        for(int j=0;j<dim_y;j++)
            tmp[i] += A[i][j]*V[j];

    for(int i=0;i<dim_y;i++)
        V[i] = tmp[i];
    free(tmp);
}

double max(double *V,int dim){
    double tmp = V[0];
    for(int i=1;i<dim;i++)
            if(fabs(V[i]) > fabs(tmp))
                tmp = V[i];
    return tmp;
}

void div_matrix(double *V,int dim,double m){
    for(int i=0;i<dim;i++)
        V[i] /= m;
}