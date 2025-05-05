%nn=[14:15,17];
%
%for n=18:18
%    f1="E:\数据集MRI\sub-pixar0"+n+"_T1w_defaced.nii";
    %f2="E:\数据集MRI\p0sub-pixar00"+n+"_T1w_defaced.nii";
%    img1=niftiread(f1);
    %img2=niftiread(f2);
    for i=1:305
        M=imread("C:\Users\linya\OneDrive\桌面\数据集\"+i+".png");
        M=mat2gray(M);
        %p=xlsread("C:\Users\linya\OneDrive\桌面\section5\一致衰减\input\"+i+"-1.xlsx");
        %str11="C:\Users\linya\OneDrive\桌面\section5\一致衰减\label\"+i+"-1.jpg";
        %str21="C:\Users\linya\OneDrive\桌面\section5\一致衰减\input\"+i+"-1.xlsx";
        %str12="C:\Users\linya\OneDrive\桌面\section5\NSW\label\"+i+"-1.jpg";
        %str22="C:\Users\linya\OneDrive\桌面\section5\NSW\input\"+i+"-1.xlsx";
        %
        %str12="C:\Users\linya\OneDrive\桌面\section5\一致衰减\label\"+i+"-2.jpg";
        %str13="C:\Users\linya\OneDrive\桌面\section5\一致衰减\label\"+i+"-3.jpg";
        %str14="C:\Users\linya\OneDrive\桌面\section5\一致衰减\label\"+i+"-4.jpg";
        str21="C:\Users\linya\OneDrive\桌面\section4\input2\"+i+"-1.xlsx";
        str22="C:\Users\linya\OneDrive\桌面\section4\input2\"+i+"-2.xlsx";
        str23="C:\Users\linya\OneDrive\桌面\section4\input2\"+i+"-3.xlsx";
        str24="C:\Users\linya\OneDrive\桌面\section4\input2\"+i+"-4.xlsx";
        %}
        %str3="E:\数据集3-2(衰减)\inputNSW\mri"+n+"-"+l+"-2.xlsx";
        %str4="E:\数据集3-2(衰减)\inputNSW\mri"+n+"-"+l+"-3.xlsx";
        %%str5="E:\数据集3-2(衰减)\inputNSW\mri"+n+"-"+l+"-4.xlsx";

        %M=zeros(256,256);
        %M=-32768*ones(256,256);

        %M2 = img1(:,50+l,:);
        %M2=reshape(M2,[176,192]);
        %M2=rot90(M2);
        %M2=mat2gray(M2);
%{
        M1 = img2(:,50+l,:);
        M1=reshape(M1,[176,192]);
        M1=rot90(M1);
        M1=mat2gray(M1);
        MM=M2-M1;
        MM=imresize(MM,[140,140]);
%}
        %M2=imresize(M2,[140,140]);
        %M(59:198,59:198)=M2;   
        %M(33:224,41:216)=M2;
        %imwrite(M,str11);
        %imwrite(M,str12);
        %imwrite(M,str13);
        %imwrite(M,str14);
%
        Nx=128;
        dx=1e-4;  % grid point spacing in the x direction [m]
        Ny=128;
        dy=1e-4;
        kgrid=kWaveGrid(Nx,dx,Ny,dy);

       %{ 
        medium.sound_speed=1500*ones(Nx,Ny);
        for i=1:140
            for j=1:140
                if MM(i,j)>0.1
                    medium.sound_speed(i+58,j+58)=3000;
                end
                if MM(i,j)<0
                    medium.sound_speed(i+58,j+58)=1550;
                end
            end
        end
        medium.density=1000*ones(Nx,Ny);
        for i=1:140
            for j=1:140
                if MM(i,j)>0.1
                    medium.density(i+58,j+58)=1900;
                end
                if MM(i,j)<0
                    medium.density(i+58,j+58)=1030;
                end
            end
        end
       %}
        medium.sound_speed=1;
        medium.density=1055;
        %medium.alpha_coeff=0.58;
        %medium.alpha_power=1.3;

        kgrid.makeTime(medium.sound_speed);%,0.6
        kgrid.Nt=640;
        
        p0=M;
        source.p0=p0;
        sensor.mask=makeCartCircle(0.44e-2,64);
        %sensor.mask = zeros(Nx, Ny);
        %sensor.mask(21, :) = 1;
        p=kspaceFirstOrder2D(kgrid,medium,source,sensor,'DataCast','gpuArray-double');
%}
        %{
        source.p0 = 0;
        sensor.time_reversal_boundary_data = p;
        p2 = kspaceFirstOrder2D(kgrid, medium, source, sensor);
        imshow(p2)
        %writematrix(p,str2);
   %}    
   %{
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
        writematrix(pa,str21);
   %}
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
        %}     
   
        %writematrix(pa,str22);
        %
        writematrix(p,str21);
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
    end

%{
for n=nn
    f1="E:\数据集MRI\sub-pixar0"+n+"_T1w_defaced.nii";
    img1=niftiread(f1);
    for l=1:75
        %str1="E:\Brain3-2一致衰减\database2\label\mri"+n+"-"+l+".jpg";
        %str2="E:\Brain3-2一致衰减\database2\input\mri"+n+"-"+l+".xlsx";
       
        M=zeros(256,256);

        M2 = img1(:,50+l,:);
        M2=reshape(M2,[176,192]);
        M2=rot90(M2);
        M2=mat2gray(M2);

        M2=imresize(M2,[140,140]);
        M(59:198,59:198)=M2;   
        %imwrite(M,str1);
        
        Nx=256;
        dx=1e-4;  % grid point spacing in the x direction [m]
        Ny=256;
        dy=1e-4;
        kgrid=kWaveGrid(Nx,dx,Ny,dy);
       
        medium.sound_speed=1;

        kgrid.makeTime(medium.sound_speed);%,0.6
        kgrid.Nt=1280;
        
        p0=M;
        source.p0=p0;
        sensor.mask=makeCartCircle(1.08e-2,128);
        p=kspaceFirstOrder2D(kgrid,medium,source,sensor,'DataCast','gpuArray-double');
          
        ht=kgrid.dt;
        t=kgrid.t_array;     
        kk=0.45;
        A=diag(exp(-kk*t));

        q=zeros(128,1280);
        for i=1:128
            for j=1:1280
                q(i,j)=sum(p(i,1:j))*ht;
            end
        end
        qa=zeros(128,1280);
        for j=1:128
            qa(j,:)=(A*q(j,:)')';
        end
        pa=zeros(128,1280);
        pa(:,1)=qa(1,:)/ht;
        for i=1:128
            X=diff(qa(i,:));
            pa(i,2:1280)=X/ht;
        end
        
        writematrix(pa,str2);
      
        clear kgrid
        clear sensor
        clear medium
        clear source
        clear p
        clear pa
        clear q
        clear qa
        clear t
    end
end
%}
