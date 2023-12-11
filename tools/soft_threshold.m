function X = soft_threshold(X,v)

    X(X>=-v & X<=v) = 0;
    X(X>v) = X(X>v) - v;
    X(X<-v) = X(X<-v) + v;
end