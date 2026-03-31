# MMA implementation based on "The method of moving asymptotes—a new method for structural optimization" by K. Svanberg (1987)

import numpy as np

def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1.0

    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(een / np.maximum(x - alfa, 1e-16), een)
    eta = np.maximum(een / np.maximum(beta - x, 1e-16), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem

        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ uxinv1 + Q @ xlinv1

        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm

        residu1 = np.concatenate([rex, rey, np.array([rez])])
        residu2 = np.concatenate([relam, rexsi, reeta, remu, np.array([rezet]), res])
        residu = np.concatenate([residu1, residu2])
        residunorm = np.sqrt(residu @ residu)
        residumax = np.max(np.abs(residu))

        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1

            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2

            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            GG = P * uxinv2[np.newaxis, :] - Q * xlinv2[np.newaxis, :]

            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam

            diagx = plam / ux3 + qlam / xl3
            diagx = 2.0 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate([blam, np.array([delz])])

                Alam = np.diag(diaglamyi) + (GG * diagxinv[np.newaxis, :]) @ GG.T
                AA = np.block([[Alam, a[:, np.newaxis]], [a[np.newaxis, :], np.array([[-zet / z]])]])

                solut = np.linalg.solve(AA, bb)
                dlam = solut[:m]
                dz = solut[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                raise NotImplementedError("This implementation expects m < n (as in topopt with one volume constraint).")

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam

            xx = np.concatenate([y, np.array([z]), lam, xsi, eta, mu, np.array([zet]), s])
            dxx = np.concatenate([dy, np.array([dz]), dlam, dxsi, deta, dmu, np.array([dzet]), ds])

            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            xold = x.copy()
            yold = y.copy()
            zold = z
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet
            sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds

                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2

                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm

                residu1 = np.concatenate([rex, rey, np.array([rez])])
                residu2 = np.concatenate([relam, rexsi, reeta, remu, np.array([rezet]), res])
                residu = np.concatenate([residu1, residu2])
                resinew = np.sqrt(residu @ residu)
                steg *= 0.5

            residunorm = resinew
            residumax = np.max(np.abs(residu))
            steg *= 2.0

        epsi *= 0.1

    return x, y, z, lam, xsi, eta, mu, zet, s


def mmasub(m, n, it, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    epsimin = 1e-5
    raa0 = 1e-5
    albefa = 0.1
    asyinit = 0.2
    asyincr = 1.1
    asydecr = 0.7

    eeen = np.ones(n)
    eeem = np.ones(m)

    low = xval - asyinit * (xmax - xmin)
    upp = xval + asyinit * (xmax - xmin)
    zzz = (xval - xold1) * (xold1 - xold2)
    factor = eeen.copy()
    factor[zzz > 0] = asyincr
    factor[zzz < 0] = asydecr

    lowmin = xval - 10.0 * (xmax - xmin)
    lowmax = xval - 0.01 * (xmax - xmin)
    uppmin = xval + 0.01 * (xmax - xmin)
    uppmax = xval + 10.0 * (xmax - xmin)
    low = np.maximum(low, lowmin)
    low = np.minimum(low, lowmax)
    upp = np.minimum(upp, uppmax)
    upp = np.maximum(upp, uppmin)

    alfa = np.maximum(low + albefa * (xval - low), xmin)
    beta = np.minimum(upp - albefa * (upp - xval), xmax)

    xmami = np.maximum(xmax - xmin, 1e-5 * eeen)
    xmamiinv = eeen / xmami

    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa0 * np.outer(eeem, xmamiinv)
    P = (P + PQ) * ux2[np.newaxis, :]
    Q = (Q + PQ) * xl2[np.newaxis, :]

    b = P @ uxinv + Q @ xlinv - fval

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def MMA_update(x, dc, fvals, dfdx, low, upp, xold1, xold2, loop, move=1.0, c_fab=500.0):
    n = x.size
    xval = x.ravel()
    df0dx = dc.ravel()

    fvals_arr = np.atleast_1d(np.asarray(fvals, dtype=float)).ravel()
    dfdx_arr = np.asarray(dfdx, dtype=float)
    if dfdx_arr.ndim == 1:
        dfdx_arr = dfdx_arr.reshape(1, -1)

    if dfdx_arr.shape[0] != fvals_arr.size or dfdx_arr.shape[1] != n:
        raise ValueError("dfdx must have shape (m, n) and match fvals length.")

    xmin = np.maximum(0.01 * np.ones(n), xval - move)
    xmax = np.minimum(np.ones(n), xval + move)

    lowv = low.ravel()
    uppv = upp.ravel()
    xold1v = xold1.ravel()
    xold2v = xold2.ravel()

    m = fvals_arr.size
    a0 = 1.0
    a = np.zeros(m)
    c = 1000.0 * np.ones(m)
    c[1] = c_fab
    d = np.zeros(m)

    xmma, _, _, _, _, _, _, _, _, low_new, upp_new = mmasub(
        m=m,
        n=n,
        it=loop,
        xval=xval,
        xmin=xmin,
        xmax=xmax,
        xold1=xold1v,
        xold2=xold2v,
        f0val=0.0,
        df0dx=df0dx,
        fval=fvals_arr,
        dfdx=dfdx_arr,
        low=lowv,
        upp=uppv,
        a0=a0,
        a=a,
        c=c,
        d=d,
    )

    return xmma.reshape(x.shape), low_new.reshape(x.shape), upp_new.reshape(x.shape)