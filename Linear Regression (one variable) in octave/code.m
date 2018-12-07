load("train.csv");
x=train(:,1);
y=train(:,2);
plot_data(x,y);
fprintf("Program Paused..press enter\n");
pause;
fprintf("Running Gradiant Discant\n");
m=length(x);
x=[ones(m, 1),x(:,1)];
theta=zeros(2,1);
iterations=15000;
alpha=0.000001;
[theta,past_cost_function]=gradiant_discent(x,y,theta,alpha,iterations,m);
theta0_vals = linspace(-100,100, 50);
theta1_vals = linspace(-100, 100, 50);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = compute_cost(x(:,2), y, t,m);
    end
end
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); 
ylabel('\theta_1');
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-500, 500, 700))
xlabel('\theta_0'); 
ylabel('\theta_1');
x_dummy=linspace(0,100,100);
line=theta(1)+theta(2).*x_dummy;
plot(line)
hold on;
plot(x(:,2),y,"r+")
figure;



