import math


class CRR:
    def __init__(self, S, K, rf, sigma, N, tipo_opzione):
        self.S = S  # prezzo S del titolo sottostante
        self.K = K  # strike price
        self.rf = rf  # tasso risk-free
        self.sigma = sigma  # volatilità implicita del titolo sottostante
        self.N = N  # numero di periodi considerati
        self.tipo_opzione = tipo_opzione  # tipo di opzione (call o put)

    def _get_u_d(self, T):
        u = math.exp(self.sigma * math.sqrt(T / self.N))
        d = math.exp(-self.sigma * math.sqrt(T / self.N))

        return u, d

    def _get_pu_pd(self, u, d):
        pu = (1 + self.rf - d) / (u - d)
        pd = 1 - pu

        return pu, pd

    def _discount_factors(self):
        discount_factor = 1 / (1 + self.rf) ** self.N

        return discount_factor

    def _get_s_next(self, u, d, St):
        Su = St * u
        Sd = St * d

        return round(Su, 4), round(Sd, 4)

    def build_tree(self, T):
        u, d = self._get_u_d(T)
        tree = [[self.S]]

        for i in range(self.N):
            list_of_su_and_sd = []
            for s in tree[-1]:
                Su, Sd = self._get_s_next(u, d, s)
                list_of_su_and_sd.append(Su)
                list_of_su_and_sd.append(Sd)
            temp_list = sorted(list(set(list_of_su_and_sd)), reverse=True)
            last = None
            for e in temp_list:
                if not last:
                    last = e
                    continue
                if abs(e - last) < 0.01:
                    temp_list.remove(e)
                last = e
            tree.append(temp_list)

        return tree

    def _get_tartaglia(self, n=None):
        if not n:
            n = self.N + 1
        if n == 0:
            return []
        elif n == 1:
            return [[1]]
        else:
            new_row = [1]
            result = self._get_tartaglia(n - 1)
            last_row = result[-1]
            for i in range(len(last_row) - 1):
                new_row.append(last_row[i] + last_row[i + 1])
            new_row += [1]
            result.append(new_row)

        return result

    def get_tree_with_probs(self, T):
        tree_with_probs = self.build_tree(T)
        u, d = self._get_u_d(T)
        pu, pd = self._get_pu_pd(u, d)
        tartaglia = self._get_tartaglia()

        for i, layer in enumerate(tree_with_probs):
            for j, s in enumerate(layer):
                total = len(layer)
                prob = pd**j * pu ** (total - j - 1) * tartaglia[i][j]
                tree_with_probs[i][j] = (tree_with_probs[i][j], prob)

        return tree_with_probs

    def _get_option_tree(self, T):
        u, d = self._get_u_d(T)
        pu, pd = self._get_pu_pd(u, d)
        tree_S = self.build_tree(T)

        tree_option = []
        list_of_values = []
        last_element = tree_S[-1]
        for price in last_element:
            if self.tipo_opzione == "Call":
                option = max([price - self.K, 0])
            elif self.tipo_opzione == "Put":
                option = max([self.K - price, 0])
            list_of_values.append(option)
        tree_option.append(list_of_values)

        return tree_option, len(tree_option[0])

    def _go_back_tree(self, T, subset_of_option_tree):
        u, d = self._get_u_d(T)
        pu, pd = self._get_pu_pd(u, d)
        lista = []
        for n, element in enumerate(subset_of_option_tree):
            if n != len(subset_of_option_tree) - 1:
                valore_precedente = (
                    pu * subset_of_option_tree[n] + pd * subset_of_option_tree[n + 1]
                ) / (1 + self.rf)
                lista.append(valore_precedente)
            else:
                continue
        return lista, len(lista)

    def build_tree_option(self, T):
        option_tree, last_layer_size = self._get_option_tree(T)

        lista = option_tree[0]
        while last_layer_size > 1:
            lista, layer = self._go_back_tree(T, lista)
            option_tree.append(lista)
            last_layer_size = layer

        return option_tree
