% --- パラメータ定義 ---
int: n_items;
int: n_groups;
int: n_cliques;
int: n_conflicts;

set of int: ITEMS = 0..n_items-1;
array[ITEMS] of int: weight0;
array[ITEMS] of int: weight1;
array[ITEMS] of int: weight2;
array[ITEMS] of int: value;
array[ITEMS] of int: group_id;

int: cap0;
int: cap1;
int: cap2;
int: bonus_val;

% クリン制約用
set of int: CLIQUES = 1..n_cliques;
array[CLIQUES] of set of int: clique_members;

% 排他ペア (アイテムID 0..n_items-1 のペア)
set of int: CONFLICTS = 1..n_conflicts;
array[CONFLICTS, 1..2] of int: conflict_pairs;

% --- 決定変数 ---
array[ITEMS] of var 0..1: x;
set of int: GROUPS = 0..n_groups-1;
array[GROUPS] of var 0..1: has_bonus;

% --- 制約条件 ---

% 1. 3次元ナップサック
constraint sum(i in ITEMS)(weight0[i] * x[i]) <= cap0;
constraint sum(i in ITEMS)(weight1[i] * x[i]) <= cap1;
constraint sum(i in ITEMS)(weight2[i] * x[i]) <= cap2;

% 2. クリン制約
constraint forall(c in CLIQUES)(
    sum(i in clique_members[c])(x[i]) <= 1
);

% 3. アイテム単位の排他制約 (0..1999のインデックスでアクセス)
constraint forall(c in CONFLICTS)(
    x[conflict_pairs[c, 1]] + x[conflict_pairs[c, 2]] <= 1
);

% 4. ボーナス判定 (3 <= 個数 <= 5 のときのみ has_bonus=1)
constraint forall(g in GROUPS)(
    let { var int: c = sum(i in ITEMS where group_id[i] == g)(x[i]) } in (
        c >= 3 * has_bonus[g] /\ 
        c <= 5 + (n_items * (1 - has_bonus[g]))
    )
);

% --- 目的関数 ---
var int: total_value = sum(i in ITEMS)(value[i] * x[i]) + sum(g in GROUPS)(bonus_val * has_bonus[g]);

solve maximize total_value;

output ["x = ", show(x), ";\n"];