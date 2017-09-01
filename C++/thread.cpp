#include<iostream>
#include<cstdlib>
#include<pthread.h>//多线程相关操作头
using namespace std;
#define NUM_THREADS 5//线程数
int sum = 0;//定义全局变量
pthread_mutex_t sum_mutex;//互斥锁
pthread_cond_t tasks_cond;//条件信号量，处理两个线程间的关系
void* say_hello(void *args){
    pthread_t pid = pthread_self();
    cout<<pid<<" hello in thread"<<*((int*)args)<<endl;
    pthread_mutex_lock(&sum_mutex);//先加锁，再修改sum的值，锁被占用就阻塞，直到拿到锁再修改sum
    cout<<"before sum is"<<sum<<"in thread"<<*((int *)args)<<endl;
    sum += *((int *)args);//修改
    cout<<"after sum is"<<sum<<"in thread"<<*((int *)args)<<endl;
    pthread_mutex_unlock(&sum_mutex);//释放锁，供其它线程使用
    //int i = *((int*)args);//强制类型转换
    //int status = 10+*((int*)args);//线程退出时添加的信息，status供主程序提取该线程的结束信息
    //pthread_exit((void*)status);
    pthread_exit(0);
}
// class Hello{
// public:
//     static void* say_hello(void* args){
//         cout<<"hello..."<<endl;
//     }
// };
int main(){
    //pthread_t tids[NUM_THREADS];//获取线程ID
    //int index[NUM_THREADS];//用来保存i的值避免被修改
    pthread_attr_t attr;//线程属性结构体，创建线程时加入的参数
    pthread_attr_init(&attr);//初始化
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);//设定线程是可连接的
    pthread_cond_init(&tasks_cond,NULL);//初始化条件信号量
    pthread_mutex_init(&tasks_mutex,NULL);//初始化互斥量
    for(int i=0;i<NUM_THREADS;++i){
        index[i]=i;
        //cout<<tids[i].x<<endl;
        int ret = pthread_create(&tids[i],NULL,say_hello,(void*)&(index[i]));//线程id,线程参数，执行函数地址，函数参数
        //cout<<"current pthread id="<<tids[i]<<endl;
        if( ret != 0){//创建成功返回0
             cout<<"pthread_create error:err_code="<<ret<<endl;
        }
    }
    pthread_attr_destroy(&attr);//释放内存
    void* status;
    for(int i=0;i<NUM_THREADS;++i){
        int ret = pthread_join(tids[i],&status);//join实现主线程和子线程同步
        if(ret!=0){
            cout<<"error code is"<<ret<<endl;
        }
        else{
            cout<<"pthread_join get status:"<<(long)status<<endl;
        }
    }
    cout<<"finally sum is"<<sum<<endl;
    pthread_mutex_destroy(&sum_mutex);
    system("pause");
    //pthread_exit(NULL);//等待各线程退出后，进程才结束，否则进程强制退出，线程处于未终止状态
}