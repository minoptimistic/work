clear all;
load("input.mat");
load("train_data.mat");
force=[];
for i=1:27
    temp =[];
    for j=1:49
        temp = [temp total_data_table(i,j+2)-2*total_data_table(i,j+1)+total_data_table(i,j)];
    end
    force = [force;temp];
end

P_train =[];
T_train =[];
a = randperm(27);
for i=1:20
    name =['a2022_3_24_',num2str(a(i))];
    temp = eval(name);
    for j=1:48
        if(j~=1)
           P_train = [P_train [temp(j,1);temp(j-1,1);temp(j,2);temp(j-1,2);total_data_table(a(i),j+1)]];
           T_train  = [T_train force(a(i),j+1)];
        else
            P_train = [P_train [temp(j,1);0;temp(j,2);0;total_data_table(a(i),j+1)]];
            T_train = [T_train force(a(i),j+1)];
        end
    end
end
P_test =[];
T_test =[];
for i=21:27
    name =['a2022_3_24_',num2str(a(i))];
    temp = eval(name);
    for j=1:48
        if(j~=1)
           P_test = [P_test [temp(j,1);temp(j-1,1);temp(j,2);temp(j-1,2);total_data_table(a(i),j+1)]];
           T_test  = [T_test force(a(i),j+1)];
        else
           P_test = [P_test [temp(j,1);0;temp(j,2);0;total_data_table(a(i),j+1)]];
           T_test = [T_test force(a(i),j+1)];
        end
    end
end
%%
n1 = min([P_train(1,:) P_test(1,:)]);n2 = min([P_train(2,:) P_test(2,:)]);n3=min([P_train(3,:) P_test(3,:)]);n4=min([P_train(4,:) P_test(4,:)]);n5=min([P_train(5,:) P_test(5,:)]);
m1 = max([P_train(1,:) P_test(1,:)]);m2 = max([P_train(2,:) P_test(2,:)]);m3=max([P_train(3,:) P_test(3,:)]);m4=max([P_train(4,:) P_test(4,:)]);m5=max([P_train(5,:) P_test(5,:)]);
no = min([T_train T_test]);mo=max([T_train T_test]);
P_train(1,:) = (P_train(1,:)-n1)/(m1-n1);P_test(1,:) = (P_test(1,:)-n1)/(m1-n1);
P_train(2,:) = (P_train(2,:)-n2)/(m2-n2);P_test(2,:) = (P_test(2,:)-n2)/(m2-n2);
P_train(3,:) = (P_train(3,:)-n3)/(m3-n3);P_test(3,:) = (P_test(3,:)-n3)/(m3-n3);
P_train(4,:) = (P_train(4,:)-n4)/(m4-n4);P_test(4,:) = (P_test(4,:)-n4)/(m4-n4);
P_train(5,:) = (P_train(5,:)-n5)/(m5-n5);P_test(5,:) = (P_test(5,:)-n5)/(m5-n5);
T_train = (T_train-no)/(mo-no);T_test= (T_test-no)/(mo-no);
%%
%BP神经网络创建训练
%创建网络
E =[];

% for i = 1:100
net = feedforwardnet(20);%newff(input,output,神经元个数)
net.trainParam.epochs = 10000; %迭代次数
net.trainParam.goal = 1e-3; %训练截止条件
net.trainParam.lr = 0.001;%学习率

net = train(net,P_train,T_train);

T_sim_bp = net(P_train);
e = (T_sim_bp - T_train).^2;
E = [E sum(e)/size(T_sim_bp,2)];
% end
% plot(1:100,E);
figure(1);
plot(1:size(T_sim_bp,2),[T_sim_bp;T_train]);
% figure(2);
% plot(T_sim_bp,T_test,'o');
% x = 0:0.1:1;
% y=@(x)x;
% hold on;
% plot(x,y(x));