
typedef bool (*funcpt)(double, double); 

bool eqcomp(double a, double b){
		return a == b;
}
bool neqcomp(double a, double b){
		return a != b;
}
bool gtcomp(double a, double b){
		return a > b;
}
bool gecomp(double a, double b){
		return a >= b;
}
bool ltcomp(double a, double b){
		return a < b;
}
bool lecomp(double a, double b){
		return a <= b;
}
