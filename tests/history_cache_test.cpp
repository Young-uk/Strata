#include "history_cache.hpp"
#include <cassert>
#include <iostream>
int main() {
 using namespace strata::history;
 std::vector<int32_t> old(2048), next;
 for (size_t i=0;i<old.size();++i) old[i]=int32_t(i+1000);
 constexpr int32_t turn=99;
 old[64]=turn;old[768]=turn;
 next.insert(next.end(),old.begin(),old.begin()+64);
 next.insert(next.end(),old.begin()+768,old.begin()+1536);
 next.push_back(20);
 auto p=match(old,next,turn);
 assert(p.prefix==64&&p.source==768&&p.length==768);
 auto unaligned=next;unaligned.insert(unaligned.begin()+4,123);
 assert(!match(old,unaligned,turn).length);
 {
   std::vector<int32_t> a(2200);for(size_t j=0;j<a.size();++j)a[j]=int32_t(j+1000);
   a[65]=a[769]=turn;a[66]=a[770]=7;a[67]=a[771]=8;
   std::vector<int32_t> b(a.begin(),a.begin()+65);b.insert(b.end(),a.begin()+769,a.begin()+1537);b.push_back(55);
   auto q=match(a,b,turn);assert(q.prefix==68&&q.source==772&&q.length==764);
 }
 auto wrong=next;wrong[0]=-1;assert(!match(old,wrong,turn).length);
 auto short_run=next;short_run.resize(100);assert(!match(old,short_run,turn).length);
 auto exact=old;exact.push_back(1);assert(!match(old,exact,turn).length);
 next.pop_back();p=match(old,next,turn);assert(p.length==764); // never sample stale logits
 assert(positions_valid({},100));assert(positions_valid({0,1,70,71},3));
 assert(!positions_valid({0,1,1,2},3));assert(!positions_valid({0,-1},1));
 assert(!positions_valid({0,1,4},2,4));assert(!positions_valid({0,1},2));
 for (size_t stride : {size_t(16),size_t(512)}) {
   std::vector<uint8_t> data(2048*stride);
   for (size_t j=0;j<data.size();++j)data[j]=uint8_t((j*13+j/stride)%251);
   auto before=data;
   assert(compact_rows(data.data(),data.size(),stride,64,768,768));
   assert(std::equal(data.begin(),data.begin()+64*stride,before.begin()));
   assert(std::equal(data.begin()+64*stride,data.begin()+832*stride,before.begin()+768*stride));
   auto moved=data;
   assert(!compact_rows(data.data(),data.size(),stride,64,2000,100));assert(data==moved);
   assert(!compact_rows(data.data(),data.size(),stride,768,64,768));assert(data==moved);
 }
 std::cout << "history matcher and position guards passed\n";
}
