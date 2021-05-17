%% setting parameters
eta=0;                %the step size of multiplier update
epsilon=5; delta=0.01;    %threshold for constraint
succ_prob=load('prob_data_48163264128192256_normal0.01_0.00025.mat','succ_prob');
succ_prob=succ_prob.succ_prob;
rate_vec=[1386 1732.5 2079 2772 3465 4158 4504.5 5197.5 6237 6756.75]';

K=length(rate_vec);    %number of rates
N=size(succ_prob,2);   %number of beamwidths
D=N*K;                 %num of arms
num_fram_pt=[1 1 1 1 1 1 1 1 1 1]';%number of transmitted frame in one time slot 

%% statistical

lambda_d=diag(rate_vec)*succ_prob;    %expectation of arms reward
lambda_d=lambda_d(:);
Lambda_d=diag(rate_vec.^2)*succ_prob.*(1-succ_prob)./num_fram_pt;    %variance of arms reward
Lambda_d=Lambda_d(:);

num_fram_pt_v=kron(ones(D,1),num_fram_pt);
rate_vec_v=kron(ones(D,1),rate_vec);
succ_prob_v=succ_prob(:);

%% algorithm

T=20e4;                %number of time slot
R_d=zeros(D,T);         %R_d(d,t) is the reward for arm d in time slot t
T_d=zeros(D,1);         %T_d(d) is the number of time that d is pulled
S_d=zeros(D,1);         %S_d(d) is the number of success for d 
Sp_d=zeros(D,1);         %Sp_d(d) is estimated success prob. for d 
a_t=zeros(T,1);      %a_t(t) is the arm pulled in time slot t

for t=1:D
    a_t(t)=t;
    [num_succ]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
    R_d(a_t(t),t)=num_succ*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
    R(t)=R_d(a_t(t),t);
    T_d(a_t(t))=T_d(a_t(t))+1; 
    S_d(a_t(t))=S_d(a_t(t))+num_succ; 
    Sp_d(a_t(t))=S_d(a_t(t))/T_d(a_t(t)); 
    con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
    
    hat_lambda(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t)))./T_d(a_t(t));
    theta(a_t(t))=hat_lambda(a_t(t));
    
end

for t=D+1:T
    for i_a=1:D
        G(i_a)=theta(i_a);
    end
    rn=rand(1);
    if rn>0.05
        ava_set=find((1-Sp_d)<delta);
        if isempty(ava_set)
            [~,a_t(t)]=max(Sp_d);
            
        else
            [~,ind]=max(G(ava_set));
            a_t(t)=ava_set(ind);
        end
    else
        a_t(t)=ceil(D*rand(1));
    end
    err_rec(t)=1-Sp_d(a_t(t));
    [num_succ]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
    R_d(a_t(t),t)=num_succ*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
    R(t)=R_d(a_t(t),t);
    T_d(a_t(t))=T_d(a_t(t))+1;  
    S_d(a_t(t))=S_d(a_t(t))+num_succ; 
    Sp_d(a_t(t))=S_d(a_t(t))/T_d(a_t(t));
    con_ind(t)=sum(R(1:t)<epsilon)-t*delta;

    hat_lambda(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t)))./T_d(a_t(t));
    for i_a=1:D
        theta(i_a)=hat_lambda(i_a);
    end
    ave_reward(t)=mean(R(1:t));
end
   
    