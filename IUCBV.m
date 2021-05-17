%% setting parameters
eta=10;                %the step size of multiplier update
zeta=2;c=2;            %para. of UCB-V
epsilon=5; delta=0.01;    %threshold for constraint
v=0;

succ_prob=load('prob_data_48163264128192256_normal0.01_0.00025.mat','succ_prob');

succ_prob=succ_prob.succ_prob;
rate_vec=[1386 1732.5 2079 2772 3465 4158 4504.5 5197.5 6237 6756.75]';

K=length(rate_vec);                   %number of rates
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
a_t=zeros(T,1);      %a_t(t) is the arm pulled in time slot t

figure(1)
for t=1:D
    a_t(t)=t;
    [num_succ]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
    R_d(a_t(t),t)=num_succ*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
    R(t)=R_d(a_t(t),t);
    T_d(a_t(t))=T_d(a_t(t))+1;  
    con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
    
    hat_lambda(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t)))./T_d(a_t(t));
    hat_Lambda(a_t(t))=sum(( R_d(a_t(t),a_t(1:t)==a_t(t)) -hat_lambda(a_t(t))).^2)/T_d(a_t(t));
    varphi(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t))<epsilon)/T_d(a_t(t));
    Phi(a_t(t))=sum((R_d(a_t(t),a_t(1:t)==a_t(t))<epsilon)-varphi(a_t(t))).^2/T_d(a_t(t));
    theta(a_t(t))=hat_lambda(a_t(t))-v*(varphi(a_t(t))-delta);
    Theta(a_t(t))=hat_Lambda(a_t(t))+v^2*Phi(a_t(t));
    v_rec(t)=v;
end

for t=D+1:T
    t
    for i_a=1:D
        G(i_a)=theta(i_a)+sqrt(2*zeta*Theta(i_a)*log(t-1)/T_d(i_a))+c*3*zeta*rate_vec_v(i_a)*log(t-1)/T_d(i_a);
    end
    [~,a_t(t)]=max(G);
    [num_succ]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
    R_d(a_t(t),t)=num_succ*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
    R(t)=R_d(a_t(t),t);
    T_d(a_t(t))=T_d(a_t(t))+1;  
    con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
    v=max(v-eta*(delta*t-(sum(R(1:t)<epsilon))),0);
    
    hat_lambda(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t)))./T_d(a_t(t));
    hat_Lambda(a_t(t))=sum(( R_d(a_t(t),a_t(1:t)==a_t(t)) -hat_lambda(a_t(t))).^2)/T_d(a_t(t));
    varphi(a_t(t))=sum(R_d(a_t(t),a_t(1:t)==a_t(t))<epsilon)/T_d(a_t(t));
    Phi(a_t(t))=sum((R_d(a_t(t),a_t(1:t)==a_t(t))<epsilon)-varphi(a_t(t))).^2/T_d(a_t(t));
    for i_a=1:D
        theta(i_a)=hat_lambda(i_a)-v*(varphi(i_a)-delta);
        Theta(i_a)=hat_Lambda(i_a)+v^2*Phi(i_a);
    end
    v_rec(t)=v;
    ave_reward(t)=mean(R(1:t));
    
end
