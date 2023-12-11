function X = X_side_info(Tmp,dim,p)
optX = Tmp(:,1:dim);
Xorth = Tmp(:,dim+1:end);
replace_dim = ceil(dim*p);
%omega = randperm(dim,replace_dim);
X = optX;
if p ~= 0
    %X(:,omega) = Xorth(:,omega);
    X(:,1:replace_dim) = Xorth(:,1:replace_dim);
end
end



% function X = new_side_info(Tmp,dim,p)
% optX = Tmp(:,1:dim);
% optX_t = optX';
% Xorth = null(optX_t);
% replace_dim = ceil(dim*p);
% omega = randperm(dim,replace_dim);
% X = optX;
% if p ~= 0
%     X(:,omega) = Xorth(:,omega);
% end
% end


