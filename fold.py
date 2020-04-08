def encode_structure_fold(fold, tree):

    def encode_node(node):
        if node.is_leaf():
            return fold.add('boxEncoder', node.box)
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder', left, right)
        elif node.is_sym():
            feature = encode_node(node.left)
            sym = node.sym
            return fold.add('symEncoder', feature, sym)

    encoding = encode_node(tree.root)
    return fold.add('sampleEncoder', encoding)


def decode_structure_fold(fold, feature, tree):
    def decode_node_box(node, feature):
        if node.is_leaf():
            box = fold.add('boxDecoder', feature)
            recon_loss = fold.add('boxLossEstimator', box, node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            return fold.add('vectorAdder', recon_loss, label_loss)
        elif node.is_adj():
            left, right = fold.add('adjDecoder', feature).split(2)
            left_loss = decode_node_box(node.left, left)
            right_loss = decode_node_box(node.right, right)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return fold.add('vectorAdder', loss, label_loss)
        elif node.is_sym():
            sym_gen, sym_param = fold.add('symDecoder', feature).split(2)
            sym_param_loss = fold.add('symLossEstimator', sym_param, node.sym)
            sym_gen_loss = decode_node_box(node.left, sym_gen)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            loss = fold.add('vectorAdder', sym_gen_loss, sym_param_loss)
            return fold.add('vectorAdder', loss, label_loss)

    feature = fold.add('sampleDecoder', feature)
    loss = decode_node_box(tree.root, feature)
    return loss
