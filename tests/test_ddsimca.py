import unittest
import numpy as np
import pandas as pd



from ddsimca import DDSIMCA, get_distparams, get_limits, process_members, process_strangers

class TestDDSIMCARes(unittest.TestCase):

    def setUp(self):
        self.calset = pd.read_csv("./demo/Target_Train.csv", index_col = 0)
        self.testset = pd.read_csv("./demo/All_Test.csv", index_col = 0)

    def test_predict_target(self):
        m = DDSIMCA("Oregano")
        m.train(self.calset, 20, True, False)

        r = m.predict(self.calset, "classic")
        r.summary()

        rt = m.predict(self.testset, "classic")
        rt.summary()

        self.assertEqual(r.ncomp, 20)
        self.assertEqual(r.center, True)
        self.assertEqual(r.scale, False)
        self.assertEqual(r.target_class, "Oregano")


class TestDDSIMCA(unittest.TestCase):

    def setUp(self):
        self.dataset = pd.read_csv("./demo/Target_Train.csv", index_col = 0)

    def test_fit(self):
        m = DDSIMCA("Oregano")
        m.train(self.dataset, 20, True, False)

        self.assertEqual(m.ncomp, 20)
        self.assertEqual(m.center, True)
        self.assertEqual(m.scale, False)
        self.assertEqual(m.target_class, "Oregano")

        # Nh classic
        Nhc = np.array([1., 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 11])
        np.testing.assert_array_almost_equal(m.hParams["classic"][1], Nhc)

        # Nh robust
        Nhr = np.array([1, 4, 6, 6, 5, 5, 5, 8, 6, 6, 9, 9, 8, 9, 9, 10, 10, 13, 10, 10])
        np.testing.assert_array_almost_equal(m.hParams["robust"][1], Nhr)

        # Nq classic
        Nqc = np.array([7, 4, 4, 3, 3, 3, 6, 6, 5, 12, 11, 12, 13, 15, 20, 18, 22, 20, 19, 18])
        np.testing.assert_array_almost_equal(m.qParams["classic"][1], Nqc)

        # Nq robust
        Nqr = np.array([9, 7, 5, 4, 3, 5, 6, 7, 8, 10, 23, 24, 10, 25, 28, 24, 25, 21, 17, 30])
        np.testing.assert_array_almost_equal(m.qParams["robust"][1], Nqr)

        # Nf classic
        np.testing.assert_array_almost_equal(m.fParams["classic"][1], Nhc + Nqc)

        # Nf robust
        np.testing.assert_array_almost_equal(m.fParams["robust"][1], Nqr + Nhr)

        # h0 classic
        np.testing.assert_array_almost_equal(
            m.hParams["classic"][0],
            np.array([0.981, 1.962, 2.942, 3.923, 4.904, 5.885, 6.865, 7.846, 8.827, 9.808, 10.788, 11.769, 12.750, 13.731, 14.712, 15.692, 16.673, 17.654, 18.635, 19.615]),
            decimal = 3
        )

        # h0 robust
        np.testing.assert_array_almost_equal(
            m.hParams["robust"][0],
            np.array([0.174, 1.195, 1.888, 3.027, 4.345, 5.326, 6.045, 6.087, 7.841, 8.629, 10.033, 10.891, 11.813, 14.061, 14.773, 15.811, 16.157, 17.819, 18.489, 19.269]),
            decimal = 3
        )

        # q0 classic
        np.testing.assert_array_almost_equal(
            m.qParams["classic"][0],
            np.array([0.715, 0.222, 0.138, 0.066, 0.043, 0.031, 0.019, 0.014, 0.010, 0.007, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002]),
            decimal = 3
        )

        # q0 robust
        np.testing.assert_array_almost_equal(
            m.qParams["robust"][0],
            np.array([0.674, 0.198, 0.135, 0.059, 0.039, 0.024, 0.018, 0.014, 0.009, 0.007, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002]),
            decimal = 3
        )


class TestProcessing(unittest.TestCase):
    def setUp(self):
        pass

    def test_process_members(self):
        f = np.array([1.0, 2., 3.1, 15.2, 10.1, 9.8, 2.2, 3.9, 9.5])
        ind = np.array([True, True, True, True, True, True, False, True, True])

        eCrit = 9.
        oCrit = 10.
        roles = np.zeros(len(f), dtype = np.int16)

        TP, FN = process_members(f, eCrit, oCrit, roles, ind)
        np.testing.assert_array_equal(roles, np.array([0, 0, 0, 2, 2, 1, 0, 0, 1]))
        self.assertEqual(TP, 4)
        self.assertEqual(FN, 4)

        roles = np.zeros(len(f), dtype = np.int16)
        ind = np.array([False, False, False, False, False, False, False, False, False])

        TP, FN = process_members(f, eCrit, oCrit, roles, ind)
        np.testing.assert_array_equal(roles, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(TP, 0)
        self.assertEqual(FN, 0)

    def test_process_strangers(self):
        f = np.array([11.0, 12., 13.1, 15.2, 10.1, 9.8, 2.2, 3.9, 29.5])
        ind = np.array([True, True, True, True, True, True, False, True, True])
        eCrit = 9.
        roles = np.zeros(len(f), dtype = np.int16)

        TN, FP, beta, s, f0, hz, Mz, Sz, k, m = process_strangers(f, 12, eCrit, roles, ind)

        #print((TN, FP, beta, s, f0, hz, Mz, Sz, k, m))

        np.testing.assert_array_equal(roles, np.array([3, 3, 3, 3, 3, 3, 0, 3, 4]))
        self.assertEqual(TN, 7)
        self.assertEqual(FP, 1)


class TestGetDistParams(unittest.TestCase):
    def setUp(self):
        self.H = np.array([
            125.4,  8.9,  0.2, 1.0,
            6.7,  0.3,  0.6, 1.0,
            17.0,  0.0,  0.2, 1.0,
            0.0,  4.7,  0.1, 1.0,
            89.9, 12.7,  0.2, 1.0,
            23.0, 15.6,  0.2, 1.0,
            0.0, 13.0,  0.0, 1.0,
            15.7,  1.5,  1.0, 1.0,
            0.0, 18.1,  0.0, 1.0,
            29.9,  5.2,  3.2, 1.0,
        ]).reshape((10, 4))


    def test_classic_estimator(self):
        """ Test classic estimator of distribution parameters """

        h0, Nh = get_distparams(self.H)
        np.testing.assert_array_almost_equal(h0, np.array([30.76, 8.00, 0.57, 1.0]))
        np.testing.assert_array_almost_equal(Nh, np.array([1., 3., 1., 250.]))

        h0, Nh = get_distparams(self.H, type = "classic")
        np.testing.assert_array_almost_equal(h0, np.array([30.76, 8.00, 0.57, 1.0]))
        np.testing.assert_array_almost_equal(Nh, np.array([1., 3., 1., 250.]))


    def test_robust_estimator(self):
        """ Test robust estimator of distribution parameters """
        h0, Nh = get_distparams(self.H, type = "robust")
        np.testing.assert_array_almost_equal(h0, np.array([23.8547017, 9.9211459, 0.3149394, 0.5033517]))
        np.testing.assert_array_almost_equal(Nh, np.array([2., 2., 2., 100.]))

    def test_getlimits(self):
        """ Test get_limits() method"""
        limE, limO = get_limits(2.1, 12.1, 0.99, 0.9999)
        self.assertAlmostEqual(limE, 4.575741, places=6)
        self.assertAlmostEqual(limO, 6.822287, places=6)


# Run the tests
if __name__ == '__main__':
    unittest.main()
