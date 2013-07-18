#include <cmath>

class Utils {
public:
  static unsigned int prev_power_of_two(unsigned int val){
    unsigned int n = 1;
    while (n*2<val){
      n*=2;
    }
    return n;
  }
};
