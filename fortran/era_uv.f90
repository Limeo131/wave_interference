        program main

        use NETCDF

!   **** read Uref, Tref, Ubar, FAWA and save 60N 10hPa time series 
!   into netCDF files ***

        integer,parameter :: kmax = 97
        integer,parameter :: nd = 91,jd = 86
        !common /irray/ qref(nd,kmax),uref(jd,kmax),tref(jd,kmax)
        !common /krray/ fawa(nd,kmax)
        !common /jrray/ ubar(nd,kmax),tbar(nd,kmax),vbar(kmax),t60(kmax)
        !common /array/ ur60n10hPa(43,12,124)
        !common /brray/ tr60n10hPa(43,12,124)
        real u60n10hPa(12,124,360,20)!(43,12,124,360,20)
        real v60n10hPa(12,124,360,20)!(43,12,124,360,20)
        !common /erray/ fw60n10hPa(43,12,124,97),wa1(43,12,124)
        !common /frray/ tf60n10hPa(43,12,124,97)
        !common /grray/ tcapn10hPa(43,12,124)
        common /hrray/ vv(360,181,97)
        common /lrray/ uu(360,181,97)
        common /mrray/ tn0(kmax),ts0(kmax),statn(kmax),stats(kmax)
        common /nrray/ wa2(43,12,124,97)
        common /prray/ tt60n(43,12,124,97)
        
        character(len=5) :: charI
        character(len=32) :: fname

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

         !fr = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGREF_N'
         !ft = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGU'
         fv = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGV'
         
         write(*,*) fv,md(n)
         
        !open(41,file =fr,  &
        !        form='unformatted',status = 'old')
        open(220,file =fv,  &
                form='unformatted',status = 'old')
       ! open(43,file =ft,  &
        !        form='unformatted',status = 'old')

          do mm = 1,md(n)*4
           

           !read(41) qref,uref,tref,fawa,ubar,tbar
           read(220) vv
           !read(43) uu
           write(*,*) vv(300,150,33)
            
           !!!! write(*,*) SIZE(uu,dim=1)
           !!!! write(*,*) SIZE(uu,dim=2)
           !!!! write(*,*) SIZE(uu,dim=3)
           
           do i = 1,360            
             do j = 141,161
              write(fy,266) m
              write(*,*) i,j,m,n,mm
                !u60n10hPa(n,mm,i,j-140) = uu(i,j,33) !100hPa-33
                v60n10hPa(n,mm,i,j-140) = vv(i,j,33)
             enddo
           enddo
           
 
! ********************************

     write(*,*)  fy,n,mm!,tt(30,151,9),tt60n(m-1978,n,mm,9)!tbar(61,65),t60(65)

! ********************************
          enddo

       !close(41)
       close(220)
       !close(43)


        enddo

       
       write(charI,'(I5)') m
      ! write(fname, '("uv_data/u100hpa_", A, ".nc")') trim(adjustl(charI))

       
      ! status = nf90_create(fname,nf90_noclobber,ncid2)
      ! status = nf90_def_dim(ncid2,"month",12,im)
      ! status = nf90_def_dim(ncid2,"time",124,it)
      ! status = nf90_def_dim(ncid2,"lon",360,ix)
      ! status = nf90_def_dim(ncid2,"lat",20,iy)
      ! status = nf90_def_var(ncid2,"u",nf90_float,   &
      !          (/iy,im,it,ix,iy/), vid2)
      ! status = nf90_put_att(ncid2,vid2,"title",'u')
      ! status = nf90_enddef(ncid2)
      ! status = nf90_put_var(ncid2,vid2,u60n10hPa)
      ! status = nf90_close(ncid2)

      ! status = nf90_open('u100hpa.nc',nf90_nowrite,ncid)
      ! status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
      ! write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
      ! status = nf90_inq_varid(ncid,"u",varID)
      ! write(6,*) 'Variable ID for U = ',varID
      ! status = nf90_get_var(ncid,varID,wa1)
      ! status = nf90_close(ncid)       

       write(fname, '("uv_data/v100hpa_", A, ".nc")') trim(adjustl(charI))
       write(6,*) fname
                                                              
       status = nf90_create(fname,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"month",12,im)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_dim(ncid2,"lon",360,ix)
       status = nf90_def_dim(ncid2,"lat",20,iy)
       status = nf90_def_var(ncid2,"v",nf90_float,   &
                (/iy,im,it,ix,iy/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",'v')
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,v60n10hPa)
       status = nf90_close(ncid2)

       status = nf90_open('v100hpa.nc',nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"v",varID)
       write(6,*) 'Variable ID for V = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)
       
       enddo 


       
        stop
        end


