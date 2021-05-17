function [reward]=arm_reward(succ_prob,num_fram_pt)
% input:
%succ_prob: success probability of frame transmission
%num_fram_pt: number of transmitted frams within one time slot

%output:
%reward: number of successfully transmitted frame

    reward=sum(rand(num_fram_pt,1)<succ_prob);

end

