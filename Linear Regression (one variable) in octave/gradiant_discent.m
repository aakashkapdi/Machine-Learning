function [theta,past_cost_function]=gradiant_discent(x,y,theta,alpha,iterations,m)
  past_cost_function=zeros(iterations,1);
  theta_0=0;
  theta_1=0;
  for i=1:iterations
    x1=x(:,2);
    past_cost_function=compute_cost(x1,y,theta,m);
    hypothesis=theta(1)+(theta(2)*x1);
    theta_0=theta_0-alpha*(1/m)*sum(hypothesis-y)
    theta_1=theta_1-alpha*(1/m)*sum((hypothesis-y).*x1)
    theta(1)=theta_0;
    theta(2)=theta_1;
  
  
  endfor
 
endfunction
