        program main

        use NETCDF

!   **** read Uref, Tref, Ubar, FAWA and save 60N 10hPa time series 
!   into netCDF files ***

        integer,parameter :: kmax = 97
        integer,parameter :: nd = 91,jd = 86
        common /irray/ qref(nd,kmax),uref(jd,kmax),tref(jd,kmax)
        common /krray/ fawa(nd,kmax)
        common /jrray/ ubar(nd,kmax),tbar(nd,kmax),vbar(33),t60(33)
        common /array/ ur60n10hPa(43,12,124,31,33)
        common /brray/ tr60n10hPa(43,12,124,31,33)
        common /crray/ ub60n10hPa(43,12,124,31,33)
        common /drray/ tb60n10hPa(43,12,124,31,33)
        common /erray/ fw60n10hPa(43,12,124,31,33),wa1(43,12,124)
        common /frray/ tf60n10hPa(43,12,124,31,33)
        common /grray/ tcapn10hPa(43,12,124)
        common /hrray/ vv(360,181,97)
        common /lrray/ tt(360,181,97)
        common /mrray/ tn0(kmax),ts0(kmax),statn(kmax),stats(kmax)
        common /nrray/ wa2(43,12,124,33)

        integer :: md(12)

        character*35 fn,fn0,fn1
        character*34 fu
        character*34 ft,fv
        character*38 fx
        character*4  fn2(12),fy,fy1,fy2
        character*19 f3
        character*37 fm
        character*38 fr

        integer :: ncid, status,nDim,nVar,nAtt,uDimID,inq
        integer :: lonID,latID,vid2,varID
        integer :: l1,l2,l3,l4,l5,l6,xtype,len,attnum

        a = 6378000.
        pi = acos(-1.)
        om = 7.29e-5
        dp = pi/180.
        dz = 500.
        h = 7000.
        r = 287.
        rkappa = r/1004.

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

        nend = 12
        if(m.eq.2021) nend = 11

        do n = 1,nend
         fr = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGREF_N'
         ft = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGT'
         fv = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGV'
         write(6,*) fr,md(n)
        open(41,file =fr,  &
                form='unformatted',status = 'old')
        open(42,file =fv,  &
                form='unformatted',status = 'old')
        open(43,file =ft,  &
                form='unformatted',status = 'old')

          do mm = 1,md(n)*4

           read(41) qref,uref,tref,fawa,ubar,tbar
           read(42) vv
           read(43) tt,tn0,ts0,statn,stats

          do j = 56,86
           ur60n10hPa(m-1978,n,mm,j-55,:) = uref(j,33:65)  !z = 100hPa-33 z = 10hPa-65
           tr60n10hPa(m-1978,n,mm,j-55,:) = tref(j,33:65)  !z = 30hpa-50 , z = 70hpa-38 , 90n - 86
          enddo
          do j = 61,91
           ub60n10hPa(m-1978,n,mm,j-60,:) = ubar(j,33:65)  !60N, z = 32 km
           tb60n10hPa(m-1978,n,mm,j-60,:) = tbar(j,33:65)  !60N, z = 32 km
           fw60n10hPa(m-1978,n,mm,j-60,:) = fawa(j,33:65)  !60N, z = 32 km
          enddo
                       ! Note for uref and tref j = 1 is 5N 
           

        !   cap = 0.    ! 60-90N polar cap average temperature
        !  do j = 61,90
        !   phi = dp*(float(j)-0.5)
        !   cap = cap + cos(phi)
        !   tcapn10hPa(m-1978,n,mm) = tcapn10hPa(m-1978,n,mm)+ &
        !   (tbar(j,65)+tbar(j+1,65))*0.5*cos(phi)
        !  enddo
        !   tcapn10hPa(m-1978,n,mm) = tcapn10hPa(m-1978,n,mm)/cap

      do j = 151,181
           t60(:) = 0.
           vbar(:) = 0.
          do i = 1,360
           t60(:) = t60(:) + tt(i,j,33:65)/360.
           vbar(:) = vbar(:) + vv(i,j,33:65)/360.
          enddo
           tf60n10hPa(m-1978,n,mm,j-150,:) = 0. 
        do i = 1,360
           tf60n10hPa(m-1978,n,mm,j-150,:) = tf60n10hPa(m-1978,n,mm,j-150,:) + &
            (vv(i,j,33:65)-vbar(:))*(tt(i,j,33:65)-t60(:))/360. 
         enddo   
           tf60n10hPa(m-1978,n,mm,j-150,:) = &
              2.*om*sin(pi/3.)*tf60n10hPa(m-1978,n,mm,j-150,:)/statn(33:65)
      enddo
 
! ********************************

     write(6,*)  fy,n,mm!,tbar(61,65),t60(65)

! ********************************
          enddo

       close(41)
       close(42)
       close(43)

        enddo
        enddo

       status = nf90_create('ubn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"ubar",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'ubn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ub60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('ubn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ubar",varID)
       write(6,*) 'Variable ID for UBAR = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ub60n10hPa(43,1,24,1,1),wa1(43,1,24)
                                                              
       status = nf90_create('urn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"uref",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'urn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ur60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('urn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"uref",varID)
       write(6,*) 'Variable ID for UREF = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ur60n10hPa(43,1,24,1,1),wa1(43,1,24)
                                                              
       status = nf90_create('trn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"tref",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'trn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,tr60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('trn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"tref",varID)
       write(6,*) 'Variable ID for TREF = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) tr60n10hPa(43,1,24,1,1),wa1(43,1,24)
                                                              
       status = nf90_create('tbn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"tbar",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'tbn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,tb60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('tbn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"tbar",varID)
       write(6,*) 'Variable ID for TBAR = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) tb60n10hPa(43,1,24,1,1),wa1(43,1,24)
                                                              
       status = nf90_create('fwn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"fawa",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'fwn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,fw60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('fwn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"fawa",varID)
       write(6,*) 'Variable ID for FAWA = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) fw60n10hPa(43,1,24,1,1),wa1(43,1,24)

                                                             
       status = nf90_create('tfn.nc',nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"year",43,iy)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lat",31,iyy)
       status = nf90_def_dim(ncid2,"height",33,iz)
       status = nf90_def_var(ncid2,"EPZ",nf90_float,   &
                (/iy,im,it,iyy,iz/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'tfn')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,tf60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('tfn.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"EPZ",varID)
       write(6,*) 'Variable ID for EPZ = ',varID
       status = nf90_get_var(ncid,varID,wa2)
       status = nf90_close(ncid)

       write(6,*) tf60n10hPa(43,1,24,1,1),wa2(43,1,24,1)
        
        stop
        end

