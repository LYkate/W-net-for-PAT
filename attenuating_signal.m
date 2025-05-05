for i=1:305
    M=imread(i+".png");
    M=mat2gray(M);
    str11="label\"+i+"-1.jpg";
    str12="label\"+i+"-2.jpg";
    str13="label\"+i+"-3.jpg";
    str14="label\"+i+"-4.jpg";
    str21="input\"+i+"-1.xlsx";
    str22="input\"+i+"-2.xlsx";
    str23="input\"+i+"-3.xlsx";
    str24="input\"+i+"-4.xlsx";
    
    imwrite(M,str11);
    imwrite(M,str12);
    imwrite(M,str13);
    imwrite(M,str14);

    Nx=128;
    dx=1e-4;  % grid point spacing in the x direction [m]
    Ny=128;
    dy=1e-4;
    kgrid=kWaveGrid(Nx,dx,Ny,dy);

    medium.sound_speed=1;  %将声速设置为1m/s

    kgrid.makeTime(medium.sound_speed);
    kgrid.Nt=640;  %设置时间节点数
        
    p0=M;
    source.p0=p0;
    sensor.mask=makeCartCircle(0.44e-2,64); %探测器
    p=kspaceFirstOrder2D(kgrid,medium,source,sensor,'DataCast','gpuArray-double');
    
%一致衰减
    ht=kgrid.dt;
    t=kgrid.t_array;     
    kk=0.45;
    A=diag(exp(-kk*t));

    q=zeros(64,640);
    for i=1:64
        for j=1:640
            q(i,j)=sum(p(i,1:j))*ht;
        end
    end
    qa=zeros(64,640);
    for j=1:64
        qa(j,:)=(A*q(j,:)')';
    end
    pa=zeros(64,640);
    pa(:,1)=qa(:,1)/ht;
    for i=1:64
        X=diff(qa(i,:));
        pa(i,2:640)=X/ht;
    end
    writematrix(pa,str21); %存储光声信号
    x=addNoise(p,20,'peak'); %添加噪声
    y=addNoise(p,25,'peak');
    z=addNoise(p,30,'peak');
    writematrix(x,str22);
    writematrix(y,str23);
    writematrix(z,str24);

%NSW衰减
%{
    kk=5/11;
    r=zeros(10,640);
    B=zeros(640,640);

    for i=1:640
        ti=t(i)-t;
        r(1,:)=sqrt(pi/2)*sign(ti);
        r(2,:)=sqrt(pi/2)*ti.*sign(ti);
        r(3,:)=sqrt(pi/2)*ti.^2.*sign(ti)/2;
        r(4,:)=sqrt(pi/2)*ti.^3.*sign(ti)/factorial(3);
        r(5,:)=sqrt(pi/2)*ti.^4.*sign(ti)/factorial(4);
        r(6,:)=sqrt(pi/2)*ti.^5.*sign(ti)/factorial(5);
        r(7,:)=sqrt(pi/2)*ti.^6.*sign(ti)/factorial(6);
        r(8,:)=sqrt(pi/2)*ti.^7.*sign(ti)/factorial(7);
        r(9,:)=sqrt(pi/2)*ti.^8.*sign(ti)/factorial(8);
        r(10,:)=sqrt(pi/2)*ti.^9.*sign(ti)/factorial(9);
        for m=1:640
            Sum=0;
            for k=1:10
                Sum=t(m)^k/factorial(k)*r(k,m)+Sum;
            end
            B(i,m)=ht/sqrt(2*pi)*exp(-kk*t(m))*Sum;
        end
    end

    A=diag(exp(-kk*t));

    q=zeros(64,640);
    for i=1:64
        for j=1:640
            q(i,j)=sum(p(i,1:j))*ht;
        end
    end
    qa=zeros(64,640);
    for j=1:64
        qa(j,:)=((A+B)*q(j,:)')';
    end

    pa=zeros(64,640);
    pa(:,1)=qa(:,1)/ht;
    for i=1:64
        X=diff(qa(i,:));
        pa(i,2:640)=X/ht;
    end
    writematrix(pa,str21);
    x=addNoise(p,20,'peak');
    y=addNoise(p,25,'peak');
    z=addNoise(p,30,'peak');
    writematrix(x,str22);
    writematrix(y,str23);
    writematrix(z,str24);
%}

    clear p
    clear kgrid
    clear sensor
    clear medium
    clear source
    clear pa
    clear qa
    clear q
end
        clear qa
        clear t
    end
end
%}
