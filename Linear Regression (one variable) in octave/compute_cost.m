function j=compute_cost(x,y,theta,m)
  j=0;
  predicted_value=theta'.*x;
  squared_errors=(predicted_value-y).^2;
  j=(1/(2*m))*sum(squared_errors(1:m));
endfunction
