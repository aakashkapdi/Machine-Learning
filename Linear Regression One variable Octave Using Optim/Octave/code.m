data=load("train.csv");
x=data(:,1);
y=data(:,2);
m=length(x);
plot_points(x,y);
f=[ones(m,1),x(:)];
pkg load optim
[p,e_var,r,p_var,y_var] = LinearRegression(f,y);
hypothesis=f*p;
plot_hypothesis(x,y,hypothesis);

