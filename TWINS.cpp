#include <bits/stdc++.h>

using namespace std;
int main()
{
    freopen("TWINS.INP","r",stdin);
    freopen("TWINS.OUT","w",stdout);
    long n;
    long k;
    int dem = 0;
    cin>>n;
    cin>>k;
    bool arr[n];
    for(int i = 2; i < n; i++)
        arr[i] = true;
    for (int i = 2; i <= ceil(sqrt(n)); i++) {
        if (arr[i]) {
            int j = i * i;
            while (j < n) {
                arr[j] = false;
                j += i;
            }
        }
    }
    for(int i = 2; i < n; i++){
        if(arr[i] && arr[i + k] && i < n && i + k < n){
            dem++;
        }
        else
            continue;
    }
    cout<<dem;
    return 0;
}
