        program main

        use NETCDF

!   **** Save QGPV in netCDF file 

        integer,parameter :: imax = 360, JMAX = 181, KMAX = 97
        common /array/ pv(imax,jmax,kmax)
        common /array/ uu(imax,jmax,kmax)
        common /array/ zz(imax,jmax,kmax)
        common /drray/ qgpv(imax,3,kmax,124)
        common /drray/ z(imax,kmax,124)
        common /drray/ u(imax,kmax,124)
        common /erray/ qgpv10(imax,jmax,124)
        common /frray/ qgz10(imax,jmax,124)
        integer :: md(12)

        character*45 fn,fn0,fn1
        character*34 fp,fu,fz,fp10,fz10
        character*34 fv
        character*4  fn2(12),fy,fy1,fy2
        character*18 f1,f2
        character*19 f3

        integer :: ncid, status,nDim,nVar,nAtt,uDimID,inq
        integer :: lonID,latID,vid2,varID
        integer :: l1,l2,l3,l4,l5,l6,xtype,len,attnum

        qgpv(:,:,:,:) = 0.

        do m = 1979,2021

        md(1) = 31
        md(2) = 28
         if(mod(m,4).eq.0) md(2) = 29
        md(3) = 31
        md(4) = 30
        md(5) = 31
        md(6) = 30
        md(7) = 31
        md(8) = 31
        md(9) = 30
        md(10) = 31
        md(11) = 30
        md(12) = 31

        fn2(1) = '_01_'
        fn2(2) = '_02_'
        fn2(3) = '_03_'
        fn2(4) = '_04_'
        fn2(5) = '_05_'
        fn2(6) = '_06_'
        fn2(7) = '_07_'
        fn2(8) = '_08_'
        fn2(9) = '_09_'
        fn2(10) = '_10_'
        fn2(11) = '_11_'
        fn2(12) = '_12_'

        write(fy,266) m
 266    format(i4)

        do n = 1,12    ! Months
         fn = '/mnt/winds/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGPV'
         fn0 = '/mnt/winds/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGU'
         fn1 = '/mnt/winds/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGZ'
         fp = fy//fn2(n)//'qgpv.nc'
         fp10 = fy//fn2(n)//'qgpv10hPa.nc'
         fu = fy//fn2(n)//'qgu.nc'
         fz = fy//fn2(n)//'qgz.nc'
         fz10 = fy//fn2(n)//'qgz10hPa.nc'
         write(6,*) fn,md(n)
        open(35,file =fn,  &
                form='unformatted',status = 'old')
        open(36,file =fn0,  &
                form='unformatted',status = 'old')
        open(37,file =fn1,  &
                form='unformatted',status = 'old')

      do mm = 1,md(n)*4

            read(35) pv 
            read(36) uu 
            read(37) zz 

            do k = 1,kmax
             qgpv(:,1,k,mm)=pv(:,150,k)     !59N
             qgpv(:,2,k,mm)=pv(:,151,k)     !60N
             qgpv(:,3,k,mm)=pv(:,152,k)     !61N
             u(:,k,mm)=uu(:,151,k)     !60N
             z(:,k,mm)=zz(:,151,k)     !60N
            enddo

             qgpv10(:,:,mm) = pv(:,:,65)
             qgz10(:,:,mm) = zz(:,:,65)

! ********************************
         write(6,*) 'file = ',mm
       enddo

       write(6,*) 'year, month =',m,n

        close(35)
        close(36)
        close(37)

       write(6,*) qgpv(1,2,50,20)

       status = nf90_create(fp,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"height",97,iz)
       status = nf90_def_dim(ncid2,"latitude",3,iy)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_var(ncid2,"qgpv",nf90_float,   &
                (/ix,iy,iz,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'qgpv.nc')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,qgpv)
       status = nf90_close(ncid2)

       qgpv(:,:,:,:) = 0.

       status = nf90_open(fp,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"qgpv",varID)
       write(6,*) 'Variable ID for QGPV = ',varID
       status = nf90_get_var(ncid,varID,qgpv)
       status = nf90_close(ncid)

       write(6,*) qgpv(1,2,50,20)

       write(6,*) u(1,50,20)

       status = nf90_create(fu,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"height",97,iz)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_var(ncid2,"qgu",nf90_float,   &
                (/ix,iz,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'qgu.nc')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,u)
       status = nf90_close(ncid2)

       u(:,:,:) = 0.

       status = nf90_open(fu,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"qgu",varID)
       write(6,*) 'Variable ID for QGU = ',varID
       status = nf90_get_var(ncid,varID,u)
       status = nf90_close(ncid)

       write(6,*) u(1,50,20)

       write(6,*) z(1,50,20)

       status = nf90_create(fz,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"height",97,iz)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_var(ncid2,"qgz",nf90_float,   &
                (/ix,iz,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'qgz.nc')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,z)
       status = nf90_close(ncid2)

       z(:,:,:) = 0.

       status = nf90_open(fz,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"qgz",varID)
       write(6,*) 'Variable ID for QGZ = ',varID
       status = nf90_get_var(ncid,varID,z)
       status = nf90_close(ncid)

       write(6,*) z(1,50,20)

       write(6,*) qgpv10(1,150,20)

       status = nf90_create(fp10,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"latitude",jmax,iy)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_var(ncid2,"qgpv10",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'qgpv10.nc')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,qgpv10)
       status = nf90_close(ncid2)

       qgpv10(:,:,:) = 0.

       status = nf90_open(fp10,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"qgpv10",varID)
       write(6,*) 'Variable ID for QGPV10 = ',varID
       status = nf90_get_var(ncid,varID,qgpv10)
       status = nf90_close(ncid)

       write(6,*) qgpv10(1,150,20)

       write(6,*) qgz10(1,150,20)

       status = nf90_create(fz10,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"latitude",jmax,iy)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_var(ncid2,"qgz10",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'qgz10.nc')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,qgz10)
       status = nf90_close(ncid2)

       qgz10(:,:,:) = 0.

       status = nf90_open(fz10,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"qgz10",varID)
       write(6,*) 'Variable ID for QGZ10 = ',varID
       status = nf90_get_var(ncid,varID,qgz10)
       status = nf90_close(ncid)

       write(6,*) qgz10(1,150,20)

       enddo
       enddo

       stop
       end
